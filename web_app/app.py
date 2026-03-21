"""
EXPoly legacy pipeline visualization: left = experimental data (3D), right = simulation (5 steps).
Experimental and simulation views share the same camera for synchronized rotation.
Run after generating data: python scripts/export_legacy_steps_for_web.py --dream3d <path> --grain-id 100 --out-dir web_app/data
For real-time grain switch: python -m uvicorn web_app.server:app --port 8050
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"

# Load run_export for real-time grain load
_run_export = None
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "export_script",
    REPO_ROOT / "scripts" / "export_legacy_steps_for_web.py",
)
if _spec and _spec.loader:
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _run_export = getattr(_mod, "run_export", None)

# For step 3: compute rotation matrix from Euler (Bunge) so axes rotate with ball
try:
    from expoly.general_func import eul2rot_bunge
except ImportError:
    eul2rot_bunge = None


def _get_current_grain_id():
    """Read grain ID that the exported data was built for."""
    p = DATA_DIR / "grain_id.txt"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None

STEP_LABELS = [
    "1. Set ball",
    "2. Ball grid to FCC",
    "3. Rotate ball",
    "4. Outer margin mesh + rotate ball",
    "5. Rough carve grain + margin meshes",
    "6. FCC atoms: grain (bulk) vs inner margin (grain-side of boundary)",
]

STEP_DESCRIPTIONS = [
    "SC grid points inside the ball (ball_struct).",
    "FCC point cloud inside the ball, not rotated (FCC_struct).",
    "FCC ball rotated by grain Euler and shifted to H center (Tr_FCC).",
    "Outer margin mesh + light rotate ball. Options: Carve grain (show blue carved atoms), Show/hide outer margin mesh.",
    "Rough carve grain (blue atoms). Options: Show inner margin mesh, Show outer margin mesh, Highlight removed atoms (rough → fine) in red.",
    "FCC atoms colored by grid region: grain (0 and 2 same color). Option: overlay experimental outer margin.",
]

STEP_FILES = [
    ["step1_ball.parquet"],
    ["step2_fcc.parquet"],
    ["step3_rotated.parquet"],
    ["step3_rotated.parquet"],
    ["step4_carved.parquet"],
    ["step5_craved_gb.parquet"],
]

# Update pipeline (inverse-box, no ball): 5 steps
UPDATE_STEP_LABELS = [
    "0. Voxels in H space",
    "1. Voxels in crystal frame",
    "2. Oriented box (crystal frame)",
    "3. Carved lattice (inverse-box, before M1)",
    "4. After M1 (grain + inner margin)",
]
UPDATE_STEP_DESCRIPTIONS = [
    "Same as experiment: H space voxels. Option: show or hide outer margin.",
    "Voxels in crystal frame (grain + outer margin). Option: show green box + FCC atoms (gray).",
    "Green box in crystal frame + voxels + gray FCC atoms (to be carved).",
    "Carved FCC grain: default crystal frame (inversed orient). Options: carved lattice in H frame (blue), inner/outer margin mesh, highlight removed (rough → carve grain).",
    "Keep only lattice points whose rounded H fall on margin-ID 0 or 2 (grain + inner margin).",
]

GRAPH_HEIGHT = "520px"
# Experimental voxels: 30% smaller than previous 8 -> ~5.6, use 5
EXPERIMENTAL_MARKER_SIZE = 5
# Step 5 FCC atoms: same size as experimental
STEP5_MARKER_SIZE = EXPERIMENTAL_MARKER_SIZE

# Unified colors: grain (0), outer margin (1), inner margin (2) — same across experiment and step 5
COLOR_GRAIN = "royalblue"
COLOR_OUTER_MARGIN = "gold"       # experiment only (margin-ID 1)
COLOR_INNER_MARGIN = "coral"     # experiment + step 5 (margin-ID 2)
# Step 1–3 and step 4 "removed" atoms: gray, same as step 4 carve option outermost
COLOR_STEP123_ATOMS = "lightgray"
STEP123_MARKER_SIZE = 2
# Atoms removed from rough carve → carve grain (step 6 highlight)
COLOR_REMOVED_ATOMS = "red"


def empty_figure():
    return go.Figure().add_annotation(
        text="No data. Run: python scripts/export_legacy_steps_for_web.py --dream3d <path> --out-dir web_app/data",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14),
    ).update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=56))


def _scene_layout():
    """System coordinate axes (x,y,z) visible; only the custom base-ball axes are hidden."""
    return dict(
        xaxis=dict(visible=True, title_font=dict(size=14), tickfont=dict(size=12)),
        yaxis=dict(visible=True, title_font=dict(size=14), tickfont=dict(size=12)),
        zaxis=dict(visible=True, title_font=dict(size=14), tickfont=dict(size=12)),
        aspectmode="data",
    )


def _transform_pts(pts: np.ndarray, R: np.ndarray | None, hcenter: np.ndarray | None) -> np.ndarray:
    """Apply p' = R @ p + hcenter. pts shape (n, 3)."""
    if R is not None:
        pts = pts @ R.T  # (n, 3) @ (3, 3)
    if hcenter is not None:
        pts = pts + np.asarray(hcenter)
    return pts


def _axes_planes_traces(
    L: float,
    R: np.ndarray | None = None,
    hcenter: np.ndarray | None = None,
    with_labels: bool = False,
) -> list:
    """Traces: origin sphere, thick X/Y/Z axes, and three semi-transparent planes (XY, YZ, ZX)."""
    # Origin: small sphere at center
    origin = np.array([[0.0, 0.0, 0.0]])
    origin = _transform_pts(origin, R, hcenter)
    traces = [
        go.Scatter3d(
            x=origin[:, 0],
            y=origin[:, 1],
            z=origin[:, 2],
            mode="markers",
            marker=dict(size=8, color="green", symbol="circle", line=dict(width=0)),
            showlegend=False,
        )
    ]
    # Thick coordinate axes: ±L along X, Y, Z (X=green, Y=gold, Z=red)
    origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    ends = np.array([[L, 0, 0], [0, L, 0], [0, 0, L]], dtype=float)
    neg_ends = np.array([[-L, 0, 0], [0, -L, 0], [0, 0, -L]], dtype=float)
    for arr in (origins, ends, neg_ends):
        arr[:] = _transform_pts(arr, R, hcenter)
    axis_colors = ["green", "gold", "red"]
    for i in range(3):
        traces.append(
            go.Scatter3d(
                x=[origins[i, 0], neg_ends[i, 0], ends[i, 0]],
                y=[origins[i, 1], neg_ends[i, 1], ends[i, 1]],
                z=[origins[i, 2], neg_ends[i, 2], ends[i, 2]],
                mode="lines",
                line=dict(color=axis_colors[i], width=10),
                showlegend=False,
            )
        )
    # Three semi-transparent planes: YZ (x=0), XZ (y=0), XY (z=0)
    quads = [
        np.array([[0, -L, -L], [0, L, -L], [0, L, L], [0, -L, L]]),   # YZ
        np.array([[-L, 0, -L], [L, 0, -L], [L, 0, L], [-L, 0, L]]),   # XZ
        np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]]),   # XY
    ]
    plane_colors = ["rgba(0,128,0,0.2)", "rgba(255,215,0,0.2)", "rgba(255,0,0,0.2)"]
    for quad, col in zip(quads, plane_colors):
        quad = _transform_pts(quad.copy(), R, hcenter)
        traces.append(
            go.Mesh3d(
                x=quad[:, 0],
                y=quad[:, 1],
                z=quad[:, 2],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                opacity=0.25,
                color=col,
                flatshading=True,
                showlegend=False,
            )
        )
    if with_labels:
        label_pts = np.array([[L, 0, 0], [0, L, 0], [0, 0, L]], dtype=float)
        label_pts = _transform_pts(label_pts, R, hcenter)
        traces.append(
            go.Scatter3d(
                x=label_pts[:, 0],
                y=label_pts[:, 1],
                z=label_pts[:, 2],
                mode="text",
                text=["100", "010", "001"],
                textfont=dict(size=14),
                showlegend=False,
            )
        )
    return traces


def _voxel_mesh_trace(df: pd.DataFrame, xc: str, yc: str, zc: str, opacity: float = 0.3, color: str = "lightblue"):
    """ConvexHull of voxel centers, then one subdivision (4 triangles per face) for a smoother mesh."""
    pts = df[[xc, yc, zc]].drop_duplicates().to_numpy()
    if len(pts) < 4:
        return None
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        new_pts_list = list(pts)
        edge_mid = {}

        def mid_idx(a, b):
            key = (min(a, b), max(a, b))
            if key not in edge_mid:
                m = (pts[a] + pts[b]) / 2
                new_pts_list.append(m)
                edge_mid[key] = len(new_pts_list) - 1
            return edge_mid[key]

        new_tri = []
        for (i, j, k) in simplices:
            mij = mid_idx(i, j)
            mjk = mid_idx(j, k)
            mki = mid_idx(k, i)
            new_tri.append((i, mij, mki))
            new_tri.append((mij, j, mjk))
            new_tri.append((mki, mjk, k))
            new_tri.append((mij, mjk, mki))
        new_pts_arr = np.array(new_pts_list)
        new_simp = np.array(new_tri)
        return go.Mesh3d(
            x=new_pts_arr[:, 0],
            y=new_pts_arr[:, 1],
            z=new_pts_arr[:, 2],
            i=new_simp[:, 0],
            j=new_simp[:, 1],
            k=new_simp[:, 2],
            opacity=opacity,
            color=color,
            flatshading=True,
        )
    except Exception:
        return None


def build_experimental_figure(show_outer_margin: bool = True) -> go.Figure:
    path = DATA_DIR / "experimental.parquet"
    if not path.exists():
        return empty_figure()
    df = pd.read_parquet(path)
    if "margin-ID" in df.columns and not show_outer_margin:
        df = df[df["margin-ID"].isin([0, 2])].copy()
    xc, yc, zc = "HX", "HY", "HZ"
    if "HX" not in df.columns:
        xc, yc, zc = "X", "Y", "Z"
    if "margin-ID" in df.columns:
        df = df.copy()
        df["region"] = df["margin-ID"].map(
            {0: "grain", 1: "outer margin", 2: "grain"}
        ).fillna("other")
        color_col = "region"
        color_map = {
            "grain": COLOR_GRAIN,
            "outer margin": COLOR_OUTER_MARGIN,
            "other": "gray",
        }
    else:
        color_col = "grain-ID" if "grain-ID" in df.columns else None
        color_map = None
    if color_col:
        df[color_col] = df[color_col].astype(str)
        fig = px.scatter_3d(
            df, x=xc, y=yc, z=zc, color=color_col, opacity=0.8,
            color_discrete_map=color_map,
        )
    else:
        fig = px.scatter_3d(df, x=xc, y=yc, z=zc, opacity=0.8)
    fig.update_traces(marker=dict(size=EXPERIMENTAL_MARKER_SIZE))
    fig.update_layout(
        scene=_scene_layout(),
        margin=dict(l=0, r=0, t=30, b=56),
        template="plotly_white",
        font=dict(size=15),
        legend=dict(
            font=dict(size=15),
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def build_step_figure(
    step_index: int,
    step6_overlay_exp_m1: bool = False,
    step4_carve_grain: bool = False,
    step4_show_outer_margin_mesh: bool = True,
    step5_show_inner_margin_mesh: bool = False,
    step5_show_outer_margin_mesh: bool = False,
    step5_highlight_removed: bool = False,
) -> go.Figure:
    files = STEP_FILES[step_index]
    traces = []

    if step_index == 3:
        # Step 4: outer margin mesh (default on) + light rotate ball. No inner margin mesh. Options: Carve grain, hide outer margin mesh.
        path_s3 = DATA_DIR / "step3_rotated.parquet"
        path_carved = DATA_DIR / "step4_carved.parquet"
        # Outer margin mesh by default (unless user unchecks)
        if step4_show_outer_margin_mesh:
            path_exp = DATA_DIR / "experimental.parquet"
            if path_exp.exists():
                df_exp = pd.read_parquet(path_exp)
                if "margin-ID" in df_exp.columns:
                    m1 = df_exp[df_exp["margin-ID"] == 1]
                    if len(m1) >= 4:
                        xc, yc, zc = "HX", "HY", "HZ"
                        if "HX" not in m1.columns:
                            xc, yc, zc = "HZ", "HY", "HX"
                        mesh_outer = _voxel_mesh_trace(m1, xc, yc, zc, opacity=0.3, color=COLOR_OUTER_MARGIN)
                        if mesh_outer is not None:
                            mesh_outer.name = "outer margin mesh"
                            traces.append(mesh_outer)
        # Rotate ball: light gray scatter (or carved vs removed if option on)
        if path_s3.exists():
            df3 = pd.read_parquet(path_s3)
            if step4_carve_grain and path_carved.exists():
                df_carved = pd.read_parquet(path_carved)
                decimals = 4
                key = lambda r: (round(r["X"], decimals), round(r["Y"], decimals), round(r["Z"], decimals))
                carved_set = set(key(df_carved.iloc[i]) for i in range(len(df_carved)))
                carved_mask = np.array([key(df3.iloc[i]) in carved_set for i in range(len(df3))])
                df_carved_pts = df3[carved_mask]
                df_rest = df3[~carved_mask]
                if len(df_carved_pts) > 0:
                    t1 = go.Scatter3d(
                        x=df_carved_pts["X"], y=df_carved_pts["Y"], z=df_carved_pts["Z"],
                        mode="markers", name="carved grain",
                        marker=dict(size=STEP5_MARKER_SIZE, color=COLOR_GRAIN, opacity=0.9),
                    )
                    traces.append(t1)
                if len(df_rest) > 0:
                    t2 = go.Scatter3d(
                        x=df_rest["X"], y=df_rest["Y"], z=df_rest["Z"],
                        mode="markers", name="rotated ball (removed)",
                        marker=dict(size=STEP123_MARKER_SIZE, color=COLOR_STEP123_ATOMS, opacity=0.25),
                    )
                    traces.append(t2)
            else:
                fig = px.scatter_3d(df3, x="X", y="Y", z="Z", opacity=0.7)
                fig.update_traces(marker=dict(size=STEP123_MARKER_SIZE, color=COLOR_STEP123_ATOMS), name="rotate ball")
                traces.extend(fig.data)
    elif step_index == 4:
        # Step 5: rough carve grain (blue atoms from step4_carved) + optional inner/outer margin meshes
        path_carved = DATA_DIR / "step4_carved.parquet"
        path_exp = DATA_DIR / "experimental.parquet"
        if path_carved.exists():
            df_c = pd.read_parquet(path_carved)
            t = go.Scatter3d(
                x=df_c["X"], y=df_c["Y"], z=df_c["Z"],
                mode="markers", name="rough carve grain",
                marker=dict(size=STEP5_MARKER_SIZE, color=COLOR_GRAIN, opacity=0.9),
            )
            traces.append(t)
        if path_exp.exists():
            df_exp = pd.read_parquet(path_exp)
            if "margin-ID" in df_exp.columns:
                xc, yc, zc = "HX", "HY", "HZ"
                if "HX" not in df_exp.columns:
                    xc, yc, zc = "HZ", "HY", "HX"
                if step5_show_inner_margin_mesh:
                    m2 = df_exp[df_exp["margin-ID"] == 2]
                    if len(m2) >= 4:
                        mesh_inner = _voxel_mesh_trace(m2, xc, yc, zc, opacity=0.35, color=COLOR_INNER_MARGIN)
                        if mesh_inner is not None:
                            mesh_inner.name = "inner margin mesh"
                            traces.append(mesh_inner)
                if step5_show_outer_margin_mesh:
                    m1 = df_exp[df_exp["margin-ID"] == 1]
                    if len(m1) >= 4:
                        mesh_outer = _voxel_mesh_trace(m1, xc, yc, zc, opacity=0.3, color=COLOR_OUTER_MARGIN)
                        if mesh_outer is not None:
                            mesh_outer.name = "outer margin mesh"
                            traces.append(mesh_outer)
        # Optional: highlight atoms removed in rough carve → carve grain (晶界处精细处理)
        if step5_highlight_removed:
            path_fine = DATA_DIR / "step5_craved_gb.parquet"
            if path_carved.exists() and path_fine.exists():
                df_rough = pd.read_parquet(path_carved)
                df_fine = pd.read_parquet(path_fine)
                decimals = 4
                key = lambda r: (round(r["X"], decimals), round(r["Y"], decimals), round(r["Z"], decimals))
                fine_set = set(key(df_fine.iloc[i]) for i in range(len(df_fine)))
                removed_mask = np.array([key(df_rough.iloc[i]) not in fine_set for i in range(len(df_rough))])
                df_removed = df_rough[removed_mask]
                if len(df_removed) > 0:
                    traces.append(
                        go.Scatter3d(
                            x=df_removed["X"], y=df_removed["Y"], z=df_removed["Z"],
                            mode="markers", name="removed (rough → fine)",
                            marker=dict(size=STEP5_MARKER_SIZE, color=COLOR_REMOVED_ATOMS, opacity=0.95),
                        )
                    )
    elif step_index == 5:
        # Step 6: FCC atoms colored by grid region – grain (bulk) vs margin
        path = DATA_DIR / "step5_craved_gb.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = df.copy()
            # margin-ID 0 and 2 both as "grain" (same color, one legend entry)
            df["region"] = df["margin-ID"].map(
                {0: "grain", 1: "outer margin", 2: "grain"}
            ).fillna("other")
            color_map = {
                "grain": COLOR_GRAIN,
                "outer margin": COLOR_OUTER_MARGIN,
                "other": "gray",
            }
            fig = px.scatter_3d(
                df, x="X", y="Y", z="Z", color="region", opacity=0.8,
                color_discrete_map=color_map,
            )
            fig.update_traces(marker=dict(size=STEP5_MARKER_SIZE))
            traces.extend(fig.data)
            # Optional: overlay experimental outer margin (ID 1) by coordinates (same grid as step 6)
            if step6_overlay_exp_m1:
                path_exp = DATA_DIR / "experimental.parquet"
                if path_exp.exists():
                    df_exp = pd.read_parquet(path_exp)
                    if "margin-ID" in df_exp.columns:
                        m1 = df_exp[df_exp["margin-ID"] == 1]
                        if len(m1) > 0:
                            traces.append(
                                go.Scatter3d(
                                    x=m1["HX"],
                                    y=m1["HY"],
                                    z=m1["HZ"],
                                    mode="markers",
                                    name="exp. outer margin",
                                    marker=dict(
                                        size=EXPERIMENTAL_MARKER_SIZE + 1,
                                        color=COLOR_OUTER_MARGIN,
                                        symbol="diamond-open",
                                        line=dict(width=1, color=COLOR_OUTER_MARGIN),
                                    ),
                                    opacity=0.9,
                                )
                            )
    else:
        for fname in files:
            path = DATA_DIR / fname
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            xc, yc, zc = "X", "Y", "Z"
            if "X" not in df.columns and "HX" in df.columns:
                xc, yc, zc = "HX", "HY", "HZ"
            color_col = None
            if "margin-ID" in df.columns:
                color_col = "margin-ID"
            elif "grain-ID" in df.columns:
                color_col = "grain-ID"
            if color_col and color_col in df.columns:
                df[color_col] = df[color_col].astype(str)
                fig = px.scatter_3d(df, x=xc, y=yc, z=zc, color=color_col, opacity=0.7)
            else:
                fig = px.scatter_3d(df, x=xc, y=yc, z=zc, opacity=0.7)
            fig.update_traces(marker=dict(size=STEP123_MARKER_SIZE, color=COLOR_STEP123_ATOMS))
            traces.extend(fig.data)
        # Coordinate axes/planes hidden for now (step 3 rotation not aligned; would confuse)
        # if step_index == 0: ...
        # elif step_index == 1: ...
        # elif step_index == 2: ...

    if not traces:
        return empty_figure()
    out = go.Figure(data=traces)
    out.update_layout(
        scene=_scene_layout(),
        margin=dict(l=0, r=0, t=30, b=56),
        template="plotly_white",
        font=dict(size=15),
        legend=dict(
            font=dict(size=15),
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
    )
    return out


def build_scale_figure(use_extend3: bool, cr: float) -> tuple[go.Figure, int | None]:
    """Build 3D figure for Scale (CR) viz: fine-carved grain colored by margin-ID. Returns (figure, n_atoms or None)."""
    prefix = "scale_extend3" if use_extend3 else "scale"
    fname = f"{prefix}_cr{cr}.parquet"
    path = DATA_DIR / fname
    if not path.exists():
        empty = go.Figure().add_annotation(
            text=f"No data: {fname}. Run export with scale precompute.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14),
        ).update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=56))
        return empty, None
    df = pd.read_parquet(path)
    n_atoms = len(df)
    df = df.copy()
    df["region"] = df["margin-ID"].map(
        {0: "grain", 1: "outer margin", 2: "grain"}
    ).fillna("other")
    color_map = {
        "grain": COLOR_GRAIN,
        "outer margin": COLOR_OUTER_MARGIN,
        "other": "gray",
    }
    fig = px.scatter_3d(
        df, x="X", y="Y", z="Z", color="region", opacity=0.8,
        color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(size=STEP5_MARKER_SIZE))
    fig.update_layout(
        scene=_scene_layout(),
        margin=dict(l=0, r=0, t=30, b=56),
        template="plotly_white",
        font=dict(size=15),
        legend=dict(
            font=dict(size=15),
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig, n_atoms


def _box_edges_trace(corners: np.ndarray, color: str = "darkgreen") -> list:
    """Corners 8x3 in order [min,min,min], [max,min,min], [max,max,min], [min,max,min], [min,min,max], ...; return 12 edge line traces."""
    # edges: (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    traces = []
    for i, j in edges:
        traces.append(
            go.Scatter3d(
                x=corners[[i, j], 0],
                y=corners[[i, j], 1],
                z=corners[[i, j], 2],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
            )
        )
    return traces


def build_update_step_figure(
    step_index: int,
    step0_show_outer: bool = True,
    step1_show_box_fcc: bool = False,
    step3_show_carved_h: bool = False,
    step3_show_inner_margin_mesh: bool = False,
    step3_show_outer_margin_mesh: bool = False,
    step3_highlight_removed: bool = False,
    step4_overlay_exp: bool = False,
) -> go.Figure:
    """Build 3D figure for Update pipeline (inverse-box) step."""
    traces = []
    xc, yc, zc = "X", "Y", "Z"

    if step_index == 0:
        # Same as experiment: H space voxels (grain + outer margin); option to hide outer margin
        path = DATA_DIR / "experimental.parquet"
        if not path.exists():
            return go.Figure().add_annotation(
                text="No data. Run: python scripts/export_update_pipeline_for_web.py --dream3d <path> --grain-id 100 --out-dir web_app/data",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14),
            ).update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=56))
        df = pd.read_parquet(path)
        if not step0_show_outer and "margin-ID" in df.columns:
            df = df[df["margin-ID"].isin([0, 2])].copy()
        hx, hy, hz = "HX", "HY", "HZ"
        if "HX" not in df.columns:
            hx, hy, hz = "HZ", "HY", "HX"
        if "margin-ID" in df.columns:
            df = df.copy()
            df["region"] = df["margin-ID"].map(
                {0: "grain", 1: "outer margin", 2: "grain"}
            ).fillna("other")
            color_map = {"grain": COLOR_GRAIN, "outer margin": COLOR_OUTER_MARGIN, "other": "gray"}
            fig = px.scatter_3d(df, x=hx, y=hy, z=hz, color="region", opacity=0.6, color_discrete_map=color_map)
        else:
            fig = px.scatter_3d(df, x=hx, y=hy, z=hz, opacity=0.6)
            fig.update_traces(marker=dict(color=COLOR_GRAIN))
        fig.update_traces(marker=dict(size=EXPERIMENTAL_MARKER_SIZE))
        traces = list(fig.data)
    elif step_index == 1:
        path = DATA_DIR / "update_step1_voxels_c.parquet"
        if not path.exists():
            return _empty_update_figure()
        df = pd.read_parquet(path)
        if "margin-ID" in df.columns:
            df = df.copy()
            df["region"] = df["margin-ID"].map(
                {0: "grain", 1: "outer margin", 2: "grain"}
            ).fillna("other")
            color_map = {"grain": COLOR_GRAIN, "outer margin": COLOR_OUTER_MARGIN, "other": "gray"}
            fig = px.scatter_3d(df, x=xc, y=yc, z=zc, color="region", opacity=0.6, color_discrete_map=color_map)
        else:
            fig = px.scatter_3d(df, x=xc, y=yc, z=zc, opacity=0.6)
            fig.update_traces(marker=dict(size=EXPERIMENTAL_MARKER_SIZE, color=COLOR_INNER_MARGIN))
        fig.update_traces(marker=dict(size=EXPERIMENTAL_MARKER_SIZE))
        traces = list(fig.data)
        if step1_show_box_fcc:
            path_c = DATA_DIR / "update_step2_box_corners_c.parquet"
            path_fcc = DATA_DIR / "update_step1_fcc_in_box_c.parquet"
            if path_c.exists():
                df_c = pd.read_parquet(path_c)
                corners = df_c[["X", "Y", "Z"]].to_numpy()
                if len(corners) >= 8:
                    traces.extend(_box_edges_trace(corners[:8], color="darkgreen"))
            if path_fcc.exists():
                df_fcc = pd.read_parquet(path_fcc)
                traces.append(
                    go.Scatter3d(
                        x=df_fcc["X"], y=df_fcc["Y"], z=df_fcc["Z"],
                        mode="markers", name="FCC in box",
                        marker=dict(size=STEP123_MARKER_SIZE, color=COLOR_STEP123_ATOMS, opacity=0.7),
                    )
                )
    elif step_index == 2:
        path_c = DATA_DIR / "update_step2_box_corners_c.parquet"
        path1 = DATA_DIR / "update_step1_voxels_c.parquet"
        path_fcc = DATA_DIR / "update_step1_fcc_in_box_c.parquet"
        if not path_c.exists():
            return _empty_update_figure()
        traces = []
        df_c = pd.read_parquet(path_c)
        corners = df_c[["X", "Y", "Z"]].to_numpy()
        if len(corners) >= 8:
            traces.extend(_box_edges_trace(corners[:8], color="darkgreen"))
        if path1.exists():
            df1 = pd.read_parquet(path1)
            traces.append(
                go.Scatter3d(
                    x=df1["X"], y=df1["Y"], z=df1["Z"],
                    mode="markers", name="voxels (crystal)",
                    marker=dict(size=EXPERIMENTAL_MARKER_SIZE, color=COLOR_INNER_MARGIN, opacity=0.6),
                )
            )
        if path_fcc.exists():
            df_fcc = pd.read_parquet(path_fcc)
            traces.append(
                go.Scatter3d(
                    x=df_fcc["X"], y=df_fcc["Y"], z=df_fcc["Z"],
                    mode="markers", name="FCC (to carve)",
                    marker=dict(size=STEP123_MARKER_SIZE, color=COLOR_STEP123_ATOMS, opacity=0.7),
                )
            )
    elif step_index == 3:
        # Default when entering Step 3 (no checkbox): crystal frame, inversed orient (first image).
        # When user checks "Carved lattice (inverse-box, before M1)": H frame, before M1 (second image), blue.
        if step3_show_carved_h:
            path = DATA_DIR / "update_step3_carved.parquet"
            name, color = "carved (H, before M1)", COLOR_GRAIN  # blue
            in_h_frame = True
        else:
            path = DATA_DIR / "update_step3_carved_c.parquet"
            name, color = "carved (crystal, inversed orient)", COLOR_INNER_MARGIN
            in_h_frame = False
        if not path.exists():
            return _empty_update_figure()
        df = pd.read_parquet(path)
        t = go.Scatter3d(
            x=df["X"], y=df["Y"], z=df["Z"],
            mode="markers", name=name,
            marker=dict(size=STEP5_MARKER_SIZE, color=color, opacity=0.85),
        )
        traces.append(t)
        # Inner / outer margin mesh (same frame as scatter: H frame → experimental; crystal → step1_voxels_c)
        if step3_show_inner_margin_mesh or step3_show_outer_margin_mesh:
            if in_h_frame:
                path_exp = DATA_DIR / "experimental.parquet"
                if path_exp.exists():
                    df_m = pd.read_parquet(path_exp)
                    xc_m, yc_m, zc_m = "HX", "HY", "HZ"
                    if "HX" not in df_m.columns:
                        xc_m, yc_m, zc_m = "HZ", "HY", "HX"
                    if step3_show_inner_margin_mesh:
                        m2 = df_m[df_m["margin-ID"] == 2]
                        if len(m2) >= 4:
                            mesh_inner = _voxel_mesh_trace(m2, xc_m, yc_m, zc_m, opacity=0.35, color=COLOR_INNER_MARGIN)
                            if mesh_inner is not None:
                                mesh_inner.name = "inner margin mesh"
                                traces.append(mesh_inner)
                    if step3_show_outer_margin_mesh:
                        m1 = df_m[df_m["margin-ID"] == 1]
                        if len(m1) >= 4:
                            mesh_outer = _voxel_mesh_trace(m1, xc_m, yc_m, zc_m, opacity=0.3, color=COLOR_OUTER_MARGIN)
                            if mesh_outer is not None:
                                mesh_outer.name = "outer margin mesh"
                                traces.append(mesh_outer)
            else:
                path1 = DATA_DIR / "update_step1_voxels_c.parquet"
                if path1.exists():
                    df_m = pd.read_parquet(path1)
                    xc_m, yc_m, zc_m = "X", "Y", "Z"
                    if step3_show_inner_margin_mesh and "margin-ID" in df_m.columns:
                        m2 = df_m[df_m["margin-ID"] == 2]
                        if len(m2) >= 4:
                            mesh_inner = _voxel_mesh_trace(m2, xc_m, yc_m, zc_m, opacity=0.35, color=COLOR_INNER_MARGIN)
                            if mesh_inner is not None:
                                mesh_inner.name = "inner margin mesh"
                                traces.append(mesh_inner)
                    if step3_show_outer_margin_mesh and "margin-ID" in df_m.columns:
                        m1 = df_m[df_m["margin-ID"] == 1]
                        if len(m1) >= 4:
                            mesh_outer = _voxel_mesh_trace(m1, xc_m, yc_m, zc_m, opacity=0.3, color=COLOR_OUTER_MARGIN)
                            if mesh_outer is not None:
                                mesh_outer.name = "outer margin mesh"
                                traces.append(mesh_outer)
        # Highlight removed atoms (rough carve → carve grain, before M1 → after M1)
        if step3_highlight_removed:
            path_rough = DATA_DIR / "update_step3_carved.parquet"
            path_after = DATA_DIR / "update_step4_after_m1.parquet"
            if path_rough.exists() and path_after.exists():
                df_rough = pd.read_parquet(path_rough)
                df_after = pd.read_parquet(path_after)
                if "HX" in df_after.columns and "HY" in df_after.columns and "HZ" in df_after.columns:
                    kept_set = set(
                        (int(df_after["HX"].iloc[i]), int(df_after["HY"].iloc[i]), int(df_after["HZ"].iloc[i]))
                        for i in range(len(df_after))
                    )
                    def key_h(r):
                        return (int(round(r["X"])), int(round(r["Y"])), int(round(r["Z"])))
                    removed_mask = np.array([key_h(df_rough.iloc[i]) not in kept_set for i in range(len(df_rough))])
                    df_removed = df_rough[removed_mask]
                    if len(df_removed) > 0:
                        if in_h_frame:
                            traces.append(
                                go.Scatter3d(
                                    x=df_removed["X"], y=df_removed["Y"], z=df_removed["Z"],
                                    mode="markers", name="removed (rough → carve grain)",
                                    marker=dict(size=STEP5_MARKER_SIZE, color=COLOR_REMOVED_ATOMS, opacity=0.95),
                                )
                            )
                        else:
                            path_tf = DATA_DIR / "update_transform.json"
                            if path_tf.exists():
                                tf = json.loads(path_tf.read_text())
                                R = np.array(tf["R"])
                                center = np.array(tf["center"])
                                pts = df_removed[["X", "Y", "Z"]].to_numpy(dtype=float)
                                pts_c = (pts - center) @ R
                                traces.append(
                                    go.Scatter3d(
                                        x=pts_c[:, 0], y=pts_c[:, 1], z=pts_c[:, 2],
                                        mode="markers", name="removed (rough → carve grain)",
                                        marker=dict(size=STEP5_MARKER_SIZE, color=COLOR_REMOVED_ATOMS, opacity=0.95),
                                    )
                                )
    elif step_index == 4:
        path = DATA_DIR / "update_step4_after_m1.parquet"
        if not path.exists():
            return _empty_update_figure()
        df = pd.read_parquet(path).copy()
        df["region"] = df["margin-ID"].map(
            {0: "grain", 1: "outer margin", 2: "grain"}
        ).fillna("other")
        color_map = {"grain": COLOR_GRAIN, "outer margin": COLOR_OUTER_MARGIN, "other": "gray"}
        fig = px.scatter_3d(df, x=xc, y=yc, z=zc, color="region", opacity=0.8, color_discrete_map=color_map)
        fig.update_traces(marker=dict(size=STEP5_MARKER_SIZE))
        traces = list(fig.data)
        if step4_overlay_exp:
            path_exp = DATA_DIR / "experimental.parquet"
            if path_exp.exists():
                df_exp = pd.read_parquet(path_exp)
                if "margin-ID" in df_exp.columns:
                    m1 = df_exp[df_exp["margin-ID"] == 1]
                    if len(m1) > 0:
                        hx, hy, hz = "HX", "HY", "HZ"
                        if "HX" not in m1.columns:
                            hx, hy, hz = "HZ", "HY", "HX"
                        traces.append(
                            go.Scatter3d(
                                x=m1[hx], y=m1[hy], z=m1[hz],
                                mode="markers", name="exp. outer margin",
                                marker=dict(
                                    size=EXPERIMENTAL_MARKER_SIZE + 1,
                                    color=COLOR_OUTER_MARGIN,
                                    symbol="diamond-open",
                                    line=dict(width=1, color=COLOR_OUTER_MARGIN),
                                ),
                                opacity=0.9,
                            )
                        )
    else:
        return _empty_update_figure()

    if not traces:
        return _empty_update_figure()
    out = go.Figure(data=traces)
    out.update_layout(
        scene=_scene_layout(),
        margin=dict(l=0, r=0, t=30, b=56),
        template="plotly_white",
        font=dict(size=15),
        legend=dict(
            font=dict(size=15),
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
    )
    return out


def _empty_update_figure():
    return go.Figure().add_annotation(
        text="No data. Run: python scripts/export_update_pipeline_for_web.py ...",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14),
    ).update_layout(template="plotly_white", margin=dict(l=0, r=0, t=30, b=56))


def _is_pipeline_path(pathname):
    return pathname in (None, "", "/", "/pipeline")


def _is_update_path(pathname):
    return pathname == "/update-pipeline"


app = dash.Dash(__name__, title="EXPoly pipeline visualization")

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.H2(id="main-title", children="EXPoly pipeline visualization", style={"textAlign": "center", "marginBottom": "8px", "fontSize": "22px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Experimental data", style={"marginBottom": "4px", "fontSize": "18px"}),
                        dcc.Graph(
                            id="left-graph",
                            style={"height": GRAPH_HEIGHT},
                            config={"scrollZoom": True},
                        ),
                        html.Div(
                            dcc.Checklist(
                                id="exp-show-outer-margin",
                                options=[{"label": " Show outer margin", "value": "on"}],
                                value=["on"],
                                style={"fontSize": "16px"},
                            ),
                            style={"marginTop": "8px"},
                        ),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "8px",
                    },
                ),
                html.Div(
                    [
                        html.H4("Simulation", style={"marginBottom": "4px", "fontSize": "18px"}),
                        html.Div(
                            [
                                dcc.Link("Pipeline (6 steps)", href="/pipeline", style={"marginRight": "16px", "fontSize": "16px"}),
                                dcc.Link("Scale (CR)", href="/scale", style={"marginRight": "16px", "fontSize": "16px"}),
                                dcc.Link("Update pipeline (inverse box)", href="/update-pipeline", style={"fontSize": "16px"}),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                        dcc.Graph(
                            id="right-graph",
                            style={"height": GRAPH_HEIGHT},
                            config={"scrollZoom": True},
                        ),
                        html.Div(
                            [
                                html.Div("Steps:", style={"fontWeight": "bold", "marginTop": "12px", "fontSize": "16px"}),
                                dcc.RadioItems(
                                    id="step-radio",
                                    options=[{"label": lab, "value": i} for i, lab in enumerate(STEP_LABELS)],
                                    value=0,
                                    style={"marginTop": "4px", "fontSize": "15px"},
                                ),
                                html.Div(
                                    dcc.Checklist(
                                        id="step4-options",
                                        options=[
                                            {"label": " Carve grain (show blue carved atoms)", "value": "carve"},
                                            {"label": " Show outer margin mesh", "value": "outer_mesh"},
                                        ],
                                        value=["outer_mesh"],
                                        style={"marginTop": "8px", "fontSize": "15px"},
                                    ),
                                    id="step4-options-container",
                                ),
                                html.Div(
                                    dcc.Checklist(
                                        id="step5-options",
                                        options=[
                                            {"label": " Inner margin mesh", "value": "inner_mesh"},
                                            {"label": " Outer margin mesh", "value": "outer_mesh"},
                                            {"label": " Highlight removed atoms (rough carve → carve grain)", "value": "removed"},
                                        ],
                                        value=[],
                                        style={"marginTop": "8px", "fontSize": "15px"},
                                    ),
                                    id="step5-options-container",
                                ),
                                html.Div(
                                    dcc.Checklist(
                                        id="step6-options",
                                        options=[
                                            {"label": " Overlay experimental outer margin by coordinates", "value": "on"},
                                        ],
                                        value=[],
                                        style={"marginTop": "8px", "fontSize": "15px"},
                                    ),
                                    id="step6-overlay-container",
                                ),
                                html.Div(id="step-description", style={"marginTop": "8px", "fontSize": "15px"}),
                            ],
                            id="pipeline-controls-container",
                        ),
                        html.Div(
                            [
                                html.Div("Extend:", style={"fontWeight": "bold", "marginTop": "12px", "fontSize": "16px"}),
                                dcc.RadioItems(
                                    id="scale-extend",
                                    options=[
                                        {"label": " No", "value": "no"},
                                        {"label": " Extend = 3", "value": "extend3"},
                                    ],
                                    value="no",
                                    style={"marginTop": "4px", "fontSize": "15px"},
                                ),
                                html.Div("CR (scale ratio):", style={"fontWeight": "bold", "marginTop": "12px", "fontSize": "16px"}),
                                dcc.RadioItems(
                                    id="scale-cr",
                                    options=[
                                        {"label": " 1.0", "value": 1.0},
                                        {"label": " 1.5", "value": 1.5},
                                        {"label": " 2.0", "value": 2.0},
                                    ],
                                    value=1.5,
                                    style={"marginTop": "4px", "fontSize": "15px"},
                                ),
                                html.Div(id="scale-description", style={"marginTop": "8px", "fontSize": "15px"}),
                            ],
                            id="scale-controls-container",
                            style={"display": "none"},
                        ),
                        html.Div(
                            [
                                html.Div("Steps:", style={"fontWeight": "bold", "marginTop": "12px", "fontSize": "16px"}),
                                dcc.RadioItems(
                                    id="update-step-radio",
                                    options=[{"label": lab, "value": i} for i, lab in enumerate(UPDATE_STEP_LABELS)],
                                    value=0,
                                    style={"marginTop": "4px", "fontSize": "15px"},
                                ),
                                html.Div(
                                    dcc.Checklist(
                                        id="update-step-options",
                                        options=[],
                                        value=[],
                                        style={"marginTop": "8px", "fontSize": "15px"},
                                    ),
                                    id="update-step-options-container",
                                    style={"marginTop": "8px", "display": "none"},
                                ),
                                html.Div(id="update-step-description", style={"marginTop": "8px", "fontSize": "15px"}),
                            ],
                            id="update-controls-container",
                            style={"display": "none"},
                        ),
                    ],
                    style={
                        "width": "48%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "8px",
                    },
                ),
            ],
            style={"width": "100%"},
        ),
        dcc.Store(id="camera-store", data=None),
        dcc.Store(id="current-grain-id", data=_get_current_grain_id()),
        dcc.Store(id="data-version", data=0),
    ],
    style={"fontFamily": "sans-serif", "padding": "12px"},
)

# Serve the same Dash app for /pipeline and /scale so direct URL / refresh works
_index_view = None
for rule in app.server.url_map.iter_rules():
    if rule.rule == "/":
        _index_view = app.server.view_functions[rule.endpoint]
        break
if _index_view is not None:
    app.server.add_url_rule("/pipeline", "serve_pipeline", _index_view)
    app.server.add_url_rule("/scale", "serve_scale", _index_view)
    app.server.add_url_rule("/update-pipeline", "serve_update_pipeline", _index_view)


@app.callback(
    Output("main-title", "children"),
    Input("url", "pathname"),
)
def update_main_title(pathname):
    if pathname == "/scale":
        return "EXPoly scale (CR) visualization"
    if pathname == "/update-pipeline":
        return "EXPoly update pipeline (inverse box)"
    return "EXPoly pipeline visualization"


@app.callback(
    Output("left-graph", "figure"),
    [
        Input("step-radio", "value"),
        Input("data-version", "data"),
        Input("exp-show-outer-margin", "value"),
    ],
    State("camera-store", "data"),
)
def update_left_figure(_, __, show_outer_values, stored_camera):
    show_outer = show_outer_values and "on" in show_outer_values
    fig = build_experimental_figure(show_outer_margin=show_outer)
    return _apply_camera(fig, stored_camera)


@app.callback(
    [
        Output("pipeline-controls-container", "style"),
        Output("scale-controls-container", "style"),
        Output("update-controls-container", "style"),
    ],
    Input("url", "pathname"),
)
def show_pipeline_or_scale_or_update_controls(pathname):
    if pathname == "/scale":
        return {"display": "none"}, {"marginTop": "8px", "fontSize": "15px"}, {"display": "none"}
    if pathname == "/update-pipeline":
        return {"display": "none"}, {"display": "none"}, {"marginTop": "8px", "fontSize": "15px"}
    return {"marginTop": "8px", "fontSize": "15px"}, {"display": "none"}, {"display": "none"}


@app.callback(
    Output("step4-options-container", "style"),
    [Input("step-radio", "value"), Input("url", "pathname")],
)
def show_step4_options_only_on_step4(step_index, pathname):
    if not _is_pipeline_path(pathname) or step_index != 3:
        return {"display": "none"}
    return {"marginTop": "8px", "fontSize": "15px"}


@app.callback(
    Output("step5-options-container", "style"),
    [Input("step-radio", "value"), Input("url", "pathname")],
)
def show_step5_options_only_on_step5(step_index, pathname):
    if not _is_pipeline_path(pathname) or step_index != 4:
        return {"display": "none"}
    return {"marginTop": "8px", "fontSize": "15px"}


@app.callback(
    Output("step6-overlay-container", "style"),
    [Input("step-radio", "value"), Input("url", "pathname")],
)
def show_step6_overlay_only_on_step6(step_index, pathname):
    if not _is_pipeline_path(pathname) or step_index != 5:
        return {"display": "none"}
    return {"marginTop": "8px", "fontSize": "15px"}


# Update pipeline: show step-specific options only under the active step
_UPDATE_STEP_OPTIONS = {
    0: [{"label": " Show outer margin", "value": "step0_outer"}],
    1: [{"label": " Show green box + FCC atoms", "value": "step1_box_fcc"}],
    3: [
        {"label": " Carved lattice (inverse-box, before M1)", "value": "step3_carved_h"},
        {"label": " Inner margin mesh", "value": "step3_inner_mesh"},
        {"label": " Outer margin mesh", "value": "step3_outer_mesh"},
        {"label": " Highlight removed atoms (rough carve → carve grain)", "value": "step3_removed"},
    ],
    4: [{"label": " Overlay experimental outer margin", "value": "step4_overlay"}],
}


@app.callback(
    [
        Output("update-step-options", "options"),
        Output("update-step-options-container", "style"),
        Output("update-step-options", "value"),
    ],
    [Input("update-step-radio", "value"), Input("url", "pathname")],
)
def update_step_options_for_step(update_step_index, pathname):
    if pathname != "/update-pipeline":
        return [], {"display": "none"}, []
    idx = update_step_index if update_step_index is not None else 0
    options = _UPDATE_STEP_OPTIONS.get(idx, [])
    if not options:
        return [], {"display": "none"}, []
    # Reset checklist value when step changes so options take effect (e.g. step 1 box+FCC)
    return options, {"marginTop": "8px"}, []


@app.callback(
    [
        Output("right-graph", "figure"),
        Output("step-description", "children"),
        Output("scale-description", "children"),
        Output("update-step-description", "children"),
    ],
    [
        Input("url", "pathname"),
        Input("step-radio", "value"),
        Input("data-version", "data"),
        Input("step4-options", "value"),
        Input("step5-options", "value"),
        Input("step6-options", "value"),
        Input("scale-extend", "value"),
        Input("scale-cr", "value"),
        Input("update-step-radio", "value"),
        Input("update-step-options", "value"),
    ],
    State("camera-store", "data"),
)
def update_right_figure(
    pathname, step_index, _, step4_values, step5_values, step6_values,
    scale_extend, scale_cr, update_step_index, update_options, stored_camera,
):
    if pathname == "/update-pipeline":
        idx = update_step_index if update_step_index is not None else 0
        opts = update_options or []
        step0_show_outer = (idx == 0) and ("step0_outer" in opts)
        step1_show_box_fcc = (idx == 1) and ("step1_box_fcc" in opts)
        # Step 3: default = crystal frame (inversed orient). Only when "Carved lattice..." is checked → H frame (before M1).
        step3_show_carved_h = (idx == 3) and ("step3_carved_h" in opts)
        step3_show_inner_mesh = (idx == 3) and ("step3_inner_mesh" in opts)
        step3_show_outer_mesh = (idx == 3) and ("step3_outer_mesh" in opts)
        step3_highlight_removed = (idx == 3) and ("step3_removed" in opts)
        step4_overlay_exp = (idx == 4) and ("step4_overlay" in opts)
        fig = build_update_step_figure(
            idx,
            step0_show_outer=step0_show_outer,
            step1_show_box_fcc=step1_show_box_fcc,
            step3_show_carved_h=step3_show_carved_h,
            step3_show_inner_margin_mesh=step3_show_inner_mesh,
            step3_show_outer_margin_mesh=step3_show_outer_mesh,
            step3_highlight_removed=step3_highlight_removed,
            step4_overlay_exp=step4_overlay_exp,
        )
        fig = _apply_camera(fig, stored_camera)
        update_desc = UPDATE_STEP_DESCRIPTIONS[idx] if 0 <= idx < len(UPDATE_STEP_DESCRIPTIONS) else ""
        return fig, "", "", update_desc
    if pathname == "/scale":
        use_extend3 = scale_extend == "extend3"
        cr = float(scale_cr) if scale_cr is not None else 1.5
        fig, n_atoms = build_scale_figure(use_extend3=use_extend3, cr=cr)
        fig = _apply_camera(fig, stored_camera)
        ext_label = "Extend = 3" if use_extend3 else "No"
        if n_atoms is not None:
            scale_desc = f"Fine-carved grain, CR = {cr}, {ext_label}. {n_atoms} atoms."
        else:
            scale_desc = f"Fine-carved grain, CR = {cr}, {ext_label}."
        return fig, "", scale_desc, ""
    if step_index is None:
        step_index = 0
    step4_opts = step4_values or []
    step4_carve = step_index == 3 and "carve" in step4_opts
    step4_show_outer_mesh = step_index == 3 and "outer_mesh" in step4_opts
    step5_opts = step5_values or []
    step5_inner_mesh = step_index == 4 and "inner_mesh" in step5_opts
    step5_outer_mesh = step_index == 4 and "outer_mesh" in step5_opts
    step5_highlight_removed = step_index == 4 and "removed" in step5_opts
    step6_opts = step6_values or []
    step6_overlay = step_index == 5 and "on" in step6_opts
    fig = build_step_figure(
        step_index,
        step6_overlay_exp_m1=step6_overlay,
        step4_carve_grain=step4_carve,
        step4_show_outer_margin_mesh=step4_show_outer_mesh,
        step5_show_inner_margin_mesh=step5_inner_mesh,
        step5_show_outer_margin_mesh=step5_outer_mesh,
        step5_highlight_removed=step5_highlight_removed,
    )
    fig = _apply_camera(fig, stored_camera)
    desc = STEP_DESCRIPTIONS[step_index] if step_index is not None else ""
    return fig, desc, "", ""


def _apply_camera(fig: go.Figure, camera: dict) -> go.Figure:
    if not fig or not camera:
        return fig
    fig = go.Figure(fig)
    fig.update_layout(scene=dict(camera=camera))
    return fig


def _camera_from_relayout(relayout: dict) -> dict | None:
    """Extract scene camera from relayoutData; support full object or partial (eye, center, up)."""
    if not relayout:
        return None
    cam = relayout.get("scene.camera")
    if isinstance(cam, dict) and ("eye" in cam or "center" in cam):
        return cam
    eye = relayout.get("scene.camera.eye")
    center = relayout.get("scene.camera.center")
    up = relayout.get("scene.camera.up")
    if eye is not None or center is not None:
        return {
            "eye": eye if eye is not None else {"x": 1.5, "y": 1.5, "z": 1.2},
            "center": center if center is not None else {"x": 0, "y": 0, "z": 0},
            "up": up if up is not None else {"x": 0, "y": 0, "z": 1},
        }
    return None


@app.callback(
    [Output("right-graph", "figure", allow_duplicate=True), Output("camera-store", "data", allow_duplicate=True)],
    Input("left-graph", "relayoutData"),
    State("right-graph", "figure"),
    State("camera-store", "data"),
    prevent_initial_call=True,
)
def sync_left_to_right(relayout, right_fig, stored_camera):
    camera = _camera_from_relayout(relayout or {})
    if not camera or not right_fig:
        return dash.no_update, stored_camera
    return _apply_camera(right_fig, camera), camera


@app.callback(
    [Output("left-graph", "figure", allow_duplicate=True), Output("camera-store", "data", allow_duplicate=True)],
    Input("right-graph", "relayoutData"),
    State("left-graph", "figure"),
    State("camera-store", "data"),
    prevent_initial_call=True,
)
def sync_right_to_left(relayout, left_fig, stored_camera):
    camera = _camera_from_relayout(relayout or {})
    if not camera or not left_fig:
        return dash.no_update, stored_camera
    return _apply_camera(left_fig, camera), camera


def _find_free_port(start: int = 8050, max_tries: int = 10) -> int:
    import socket
    for i in range(max_tries):
        port = start + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return start


if __name__ == "__main__":
    port = _find_free_port(8050)
    if port != 8050:
        print(f"Port 8050 in use; using port {port}")
    app.run_server(debug=True, port=port)
