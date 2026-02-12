# src/expoly/generate_voronoi.py
"""
Extract GB surface / triple line / quadruple point topology from a LAMMPS dump,
build Voronoi mesh (pair_to_mesh), voxelize grains, and output voxel_all.csv
with 0-based integer grid coordinates.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

logger = logging.getLogger(__name__)

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None
    logger.warning("sklearn not available; pair_to_mesh building may fail")

PathLike = Union[str, Path]


def read_lammps_dump_one_timestep(path: PathLike) -> pd.DataFrame:
    """Read one timestep of LAMMPS dump. Returns DataFrame with atom-ID, type, X, Y, Z, grain-ID."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms_line_idx = None
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            atoms_line_idx = i
            break
    if atoms_line_idx is None:
        raise RuntimeError("No 'ITEM: ATOMS' line found")
    header_tokens = lines[atoms_line_idx].strip().split()
    colnames = header_tokens[2:]
    data_lines = []
    for line in lines[atoms_line_idx + 1 :]:
        if line.startswith("ITEM:"):
            break
        if line.strip() == "":
            continue
        data_lines.append(line)
    data_str = "\n".join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep=r"\s+", header=None, names=colnames)
    rename_dict = {}
    if "id" in df.columns:
        rename_dict["id"] = "atom-ID"
    if "grainid" in df.columns:
        rename_dict["grainid"] = "grain-ID"
    if "x" in df.columns:
        rename_dict["x"] = "X"
    if "y" in df.columns:
        rename_dict["y"] = "Y"
    if "z" in df.columns:
        rename_dict["z"] = "Z"
    df = df.rename(columns=rename_dict)
    logger.info("[read] atoms=%d, columns=%s", len(df), list(df.columns))
    return df


def classify_topology_fast(
    DATA: pd.DataFrame,
    k: int = 13,
    min_other_atoms: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, List[int]]]:
    """Classify atoms as surface/line/point using k-NN. Returns surface_all, line_all, point_all, outer_map."""
    pos = DATA[["X", "Y", "Z"]].to_numpy()
    grain = DATA["grain-ID"].to_numpy().astype(np.intp)
    atom_id = DATA["atom-ID"].to_numpy().astype(np.intp)
    N = len(DATA)
    tree = cKDTree(pos)
    _, idx_knn = tree.query(pos, k=k)
    surface_idx: List[int] = []
    line_idx: List[int] = []
    point_idx: List[int] = []
    outer_surface: Dict[int, List[int]] = {}
    outer_line: Dict[int, List[int]] = {}
    outer_point: Dict[int, List[int]] = {}
    for i in range(N):
        neigh_idx = idx_knn[i]
        neigh_grain = grain[neigh_idx]
        mask_other = neigh_grain != grain[i]
        if np.sum(mask_other) < min_other_atoms:
            continue
        unique_grains = np.unique(neigh_grain)
        atom_pos_type = len(unique_grains)
        if atom_pos_type == 1:
            continue
        outer = unique_grains[unique_grains != grain[i]].tolist()
        a_id = int(atom_id[i])
        if atom_pos_type == 2:
            surface_idx.append(i)
            outer_surface[a_id] = outer
        elif atom_pos_type == 3:
            line_idx.append(i)
            outer_line[a_id] = outer
        elif atom_pos_type == 4:
            point_idx.append(i)
            outer_point[a_id] = outer
    surface_all = DATA.iloc[surface_idx].copy()
    line_all = DATA.iloc[line_idx].copy()
    point_all = DATA.iloc[point_idx].copy()
    if surface_idx:
        surf_outer_df = (
            pd.DataFrame.from_dict(outer_surface, orient="index")
            .reset_index()
            .rename(columns={"index": "atom-ID", 0: "outer1"})
        )
        surface_all = surface_all.merge(surf_outer_df, on="atom-ID", how="left")
    if line_idx:
        line_outer_df = (
            pd.DataFrame.from_dict(outer_line, orient="index")
            .reset_index()
            .rename(columns={"index": "atom-ID", 0: "outer1", 1: "outer2"})
        )
        line_all = line_all.merge(line_outer_df, on="atom-ID", how="left")
    if point_idx:
        point_outer_df = (
            pd.DataFrame.from_dict(outer_point, orient="index")
            .reset_index()
            .rename(columns={"index": "atom-ID", 0: "outer1", 1: "outer2", 2: "outer3"})
        )
        point_all = point_all.merge(point_outer_df, on="atom-ID", how="left")
    outer_map: Dict[int, List[int]] = {}
    outer_map.update(outer_surface)
    outer_map.update(outer_line)
    outer_map.update(outer_point)
    logger.info(
        "[classify] surface=%d, line=%d, point=%d",
        len(surface_all),
        len(line_all),
        len(point_all),
    )
    return surface_all, line_all, point_all, outer_map


def assign_outer_shell_to_face_grains(
    data_mod: pd.DataFrame,
    mask_outer: pd.Series,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
) -> Tuple[int, int, int, int, int, int]:
    """Assign outer-shell atoms to 6 face grain IDs. Returns G_xmin, G_xmax, G_ymin, G_ymax, G_zmin, G_zmax."""
    grain_max = int(data_mod["grain-ID"].max())
    face_grain_ids = np.array(
        [grain_max + 1, grain_max + 2, grain_max + 3, grain_max + 4, grain_max + 5, grain_max + 6],
        dtype=np.intp,
    )
    outer = data_mod.loc[mask_outer, ["X", "Y", "Z"]]
    x = outer["X"].to_numpy()
    y = outer["Y"].to_numpy()
    z = outer["Z"].to_numpy()
    dists = np.vstack(
        [
            x - xmin,
            xmax - x,
            y - ymin,
            ymax - y,
            z - zmin,
            zmax - z,
        ]
    ).T
    face_idx = np.argmin(dists, axis=1)
    data_mod.loc[mask_outer, "grain-ID"] = face_grain_ids[face_idx]
    return tuple(int(gid) for gid in face_grain_ids)


def build_quad_unique(point_all: pd.DataFrame) -> pd.DataFrame:
    """Merge quadruple points by quad_id; aggregate X,Y,Z by mean."""
    if not {"outer1", "outer2", "outer3"}.issubset(point_all.columns):
        raise RuntimeError("point_all must have outer1, outer2, outer3 columns")
    point_all = point_all.copy()

    def make_quad_id(row: pd.Series) -> Tuple[int, ...]:
        return tuple(
            sorted(
                [
                    int(row["grain-ID"]),
                    int(row["outer1"]),
                    int(row["outer2"]),
                    int(row["outer3"]),
                ]
            )
        )

    point_all["quad_id"] = point_all.apply(make_quad_id, axis=1)
    quad_group = (
        point_all.groupby("quad_id", as_index=False)
        .agg(
            {
                "X": "mean",
                "Y": "mean",
                "Z": "mean",
                "grain-ID": "first",
                "outer1": "first",
                "outer2": "first",
                "outer3": "first",
            }
        )
    )
    quad_unique = quad_group.rename(
        columns={"X": "X_mean", "Y": "Y_mean", "Z": "Z_mean"}
    )
    quad_unique["quad_set"] = quad_unique["quad_id"].apply(lambda t: frozenset(t))
    return quad_unique


def build_triple_line_segments(
    line_all: pd.DataFrame,
    quad_unique: pd.DataFrame,
) -> pd.DataFrame:
    """Build triple-line segments connecting quadruple junctions."""
    if not {"outer1", "outer2"}.issubset(line_all.columns):
        raise RuntimeError("line_all must have outer1, outer2 columns")
    line_all = line_all.copy()

    def make_tri_id(row: pd.Series) -> Tuple[int, ...]:
        return tuple(sorted([int(row["grain-ID"]), int(row["outer1"]), int(row["outer2"])]))

    line_all["tri_id"] = line_all.apply(make_tri_id, axis=1)
    tri_unique = line_all["tri_id"].unique()
    quad_sets = list(quad_unique["quad_set"])
    segments: List[Tuple[int, int]] = []
    for tri in tri_unique:
        tri_set = set(tri)
        incident_quads = [qi for qi, qset in enumerate(quad_sets) if tri_set.issubset(qset)]
        if len(incident_quads) < 2:
            continue
        for i, j in itertools.combinations(incident_quads, 2):
            segments.append((i, j))
    return pd.DataFrame(segments, columns=["quad_i", "quad_j"])


def connected_components_on_pair_nodes(node_idx: List[int]) -> List[List[int]]:
    """Find connected components among nodes (by distance < threshold)."""
    if not node_idx:
        return []
    node_set = set(node_idx)
    visited = set()
    comps = []
    for start in node_idx:
        if start in visited:
            continue
        stack = [start]
        comp = []
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.append(n)
            for other in node_set:
                if other not in visited:
                    stack.append(other)
        if comp:
            comps.append(comp)
    return comps


def build_pair_to_mesh(
    surface_all: pd.DataFrame,
    line_all: pd.DataFrame,
    quad_unique: pd.DataFrame,
    segments_df: pd.DataFrame,
    voxel_size: float = 2.0,
) -> Dict[Tuple[int, int], List[Dict]]:
    """
    Build pair_to_mesh from quad_unique + segments: for each grain pair (g1,g2),
    find incident junctions, triangulate, return dict[(g1,g2)] = [patches].
    """
    if PCA is None:
        raise RuntimeError("sklearn.decomposition.PCA required for pair_to_mesh building")
    quad_ids = list(quad_unique["quad_id"])
    quad_pos = quad_unique[["X_mean", "Y_mean", "Z_mean"]].to_numpy()
    pair_to_mesh: Dict[Tuple[int, int], List[Dict]] = {}
    pairs_to_process = set()
    for _, row in surface_all.iterrows():
        g1, g2 = int(row["grain-ID"]), int(row["outer1"])
        pairs_to_process.add(tuple(sorted([g1, g2])))
    for pair in pairs_to_process:
        g1, g2 = pair
        node_idx = [
            idx
            for idx, q in enumerate(quad_ids)
            if {g1, g2}.issubset(set(int(x) for x in q))
        ]
        if len(node_idx) < 3:
            continue
        comps = connected_components_on_pair_nodes(node_idx)
        comps = [c for c in comps if len(c) >= 3]
        if not comps:
            continue
        comps.sort(key=len, reverse=True)
        main_comp = comps[0]
        pts = quad_pos[main_comp]
        if pts.shape[0] < 3:
            continue
        pca = PCA(n_components=3)
        pca.fit(pts)
        center = pts.mean(axis=0)
        v1 = pca.components_[0]
        v2 = pca.components_[1]
        diff = pts - center
        u = diff @ v1
        v = diff @ v2
        uv = np.stack([u, v], axis=1)
        try:
            tri = Delaunay(uv)
        except Exception as e:
            logger.warning(f"Delaunay failed for pair {pair}: {e}")
            continue
        simplices = tri.simplices
        if simplices.shape[0] == 0:
            continue
        patch = {
            "verts": pts,
            "simplices": simplices,
            "i": simplices[:, 0],
            "j": simplices[:, 1],
            "k": simplices[:, 2],
        }
        pair_to_mesh[pair] = [patch]
    logger.info("[pair_to_mesh] built %d pairs", len(pair_to_mesh))
    return pair_to_mesh


def collect_grain_planes(
    gid: int,
    pair_to_mesh_shift: Dict[Tuple[int, int], List[Dict]],
    grain_centers: Dict[int, np.ndarray],
) -> List[Dict]:
    """Collect plane constraints for a grain from pair_to_mesh_shift."""
    if gid not in grain_centers:
        return []
    c = grain_centers[gid]
    planes_g = []
    for (g1, g2), patches in pair_to_mesh_shift.items():
        if gid not in (g1, g2):
            continue
        verts_all = np.concatenate([np.asarray(p["verts"], dtype=float) for p in patches], axis=0)
        if verts_all.shape[0] < 3:
            continue
        p0 = verts_all.mean(axis=0)
        if PCA is None:
            continue
        pca = PCA(n_components=3)
        pca.fit(verts_all)
        n_raw = pca.components_[2]
        s_center = np.dot(c - p0, n_raw)
        side = ">=" if s_center >= 0 else "<="
        planes_g.append({"p0": p0, "n": n_raw, "side": side})
    return planes_g


def voxelize_grain_on_integer_grid(
    gid: int,
    pair_to_mesh_shift: Dict[Tuple[int, int], List[Dict]],
    grain_centers: Dict[int, np.ndarray],
    voxel_size: float,
    S_min: np.ndarray,
    index_step: int = 1,
    margin_nvox: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Voxelize a grain using half-space intersection. Returns (idx_in, centers_in) or (None, None).
    """
    planes_g = collect_grain_planes(gid, pair_to_mesh_shift, grain_centers)
    if not planes_g:
        logger.debug(f"[voxelize] grain {gid} has no planes")
        return None, None
    verts_all = []
    for (g1, g2), patches in pair_to_mesh_shift.items():
        if gid not in (g1, g2):
            continue
        for p in patches:
            verts_all.append(np.asarray(p["verts"], dtype=float))
    if not verts_all:
        logger.debug(f"[voxelize] grain {gid} has no mesh vertices")
        return None, None
    verts_all = np.concatenate(verts_all, axis=0)
    vmin = verts_all.min(axis=0)
    vmax = verts_all.max(axis=0)
    ix_min_f = (vmin[0] - S_min[0]) / voxel_size - 0.5
    iy_min_f = (vmin[1] - S_min[1]) / voxel_size - 0.5
    iz_min_f = (vmin[2] - S_min[2]) / voxel_size - 0.5
    ix_max_f = (vmax[0] - S_min[0]) / voxel_size - 0.5
    iy_max_f = (vmax[1] - S_min[1]) / voxel_size - 0.5
    iz_max_f = (vmax[2] - S_min[2]) / voxel_size - 0.5

    def align_range(i_min: float, i_max: float, step: int) -> np.ndarray:
        if step <= 1:
            return np.arange(int(np.floor(i_min)) - margin_nvox, int(np.ceil(i_max)) + margin_nvox + 1, 1, dtype=int)
        start = ((int(np.floor(i_min)) - margin_nvox) + step - 1) // step * step
        end = (int(np.ceil(i_max)) + margin_nvox) // step * step
        if start > end:
            return np.array([], dtype=int)
        return np.arange(start, end + 1, step, dtype=int)

    ix_vals = align_range(ix_min_f, ix_max_f, index_step)
    iy_vals = align_range(iy_min_f, iy_max_f, index_step)
    iz_vals = align_range(iz_min_f, iz_max_f, index_step)
    if ix_vals.size == 0 or iy_vals.size == 0 or iz_vals.size == 0:
        logger.debug(f"[voxelize] grain {gid} index range empty")
        return None, None
    IX, IY, IZ = np.meshgrid(ix_vals, iy_vals, iz_vals, indexing="ij")
    idx_all = np.stack([IX.ravel(), IY.ravel(), IZ.ravel()], axis=1)
    centers_all = S_min + (idx_all.astype(float) + 0.5) * voxel_size
    inside = np.ones(len(idx_all), dtype=bool)
    for pl in planes_g:
        p0 = pl["p0"]
        n = pl["n"]
        side = pl["side"]
        s = (centers_all - p0) @ n
        if side == "<=":
            inside &= s <= 0
        elif side == ">=":
            inside &= s >= 0
    idx_in = idx_all[inside]
    centers_in = centers_all[inside]
    logger.debug(f"[voxelize] grain {gid}: candidates={len(idx_all)}, inside={len(idx_in)}")
    return idx_in, centers_in


def map_to_full_grid(
    voxel_all: pd.DataFrame,
    voxel_size: float,
) -> pd.DataFrame:
    """
    Map voxel_all to 0-based full rectangular grid (fill holes with nearest neighbor).
    Returns voxel_all with updated voxel-X/Y/Z coordinates.
    """
    step = int(voxel_size)
    voxel_all = voxel_all.drop_duplicates(
        subset=["voxel-X", "voxel-Y", "voxel-Z"],
        keep="first",
    ).reset_index(drop=True)
    ix_raw = (voxel_all["voxel-X"] // step).astype(int).to_numpy()
    iy_raw = (voxel_all["voxel-Y"] // step).astype(int).to_numpy()
    iz_raw = (voxel_all["voxel-Z"] // step).astype(int).to_numpy()
    pts_exist_raw = np.column_stack([ix_raw, iy_raw, iz_raw])
    ix_min, iy_min, iz_min = pts_exist_raw.min(axis=0)
    ix_max, iy_max, iz_max = pts_exist_raw.max(axis=0)
    logger.info(
        "原始 index 范围: ix=%d→%d, iy=%d→%d, iz=%d→%d",
        ix_raw.min(),
        ix_raw.max(),
        iy_raw.min(),
        iy_raw.max(),
        iz_raw.min(),
        iz_raw.max(),
    )
    ix0 = ix_raw - ix_min
    iy0 = iy_raw - iy_min
    iz0 = iz_raw - iz_min
    pts_exist = np.column_stack([ix0, iy0, iz0])
    nx = ix0.max() + 1
    ny = iy0.max() + 1
    nz = iz0.max() + 1
    logger.info("0-based 网格尺寸: nx=%d, ny=%d, nz=%d", nx, ny, nz)
    grid_pts = np.array(
        list(itertools.product(range(nx), range(ny), range(nz))),
        dtype=np.intp,
    )
    tree_exist = cKDTree(pts_exist)
    _, indices_src = tree_exist.query(grid_pts, k=1)
    voxel_full = voxel_all.iloc[indices_src].copy()
    voxel_full["voxel-X"] = (grid_pts[:, 0] * step).astype(int)
    voxel_full["voxel-Y"] = (grid_pts[:, 1] * step).astype(int)
    voxel_full["voxel-Z"] = (grid_pts[:, 2] * step).astype(int)
    logger.info("完整长方体网格点数=%d, 原始 voxel 数=%d", len(grid_pts), len(voxel_all))
    return voxel_full


def run(
    dump_path: PathLike,
    output_path: PathLike,
    crop_ratio: float = 0.015,
    k: int = 25,
    min_other_atoms: int = 4,
    voxel_size: float = 2.0,
) -> Path:
    """
    Full pipeline: read dump → crop → classify → build mesh → voxelize → map to grid → write voxel_all.csv.
    Returns output_path.
    """
    dump_path = Path(dump_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    DATA = read_lammps_dump_one_timestep(dump_path)
    xmin, xmax = float(DATA["X"].min()), float(DATA["X"].max())
    ymin, ymax = float(DATA["Y"].min()), float(DATA["Y"].max())
    zmin, zmax = float(DATA["Z"].min()), float(DATA["Z"].max())
    Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin
    x_in_min = xmin + crop_ratio * Lx
    x_in_max = xmax - crop_ratio * Lx
    y_in_min = ymin + crop_ratio * Ly
    y_in_max = ymax - crop_ratio * Ly
    z_in_min = zmin + crop_ratio * Lz
    z_in_max = zmax - crop_ratio * Lz
    mask_inner = (
        (DATA["X"] >= x_in_min) & (DATA["X"] <= x_in_max)
        & (DATA["Y"] >= y_in_min) & (DATA["Y"] <= y_in_max)
        & (DATA["Z"] >= z_in_min) & (DATA["Z"] <= z_in_max)
    )
    mask_outer = ~mask_inner
    logger.info("atoms: total=%d, inner=%d, outer=%d", len(DATA), mask_inner.sum(), mask_outer.sum())

    DATA_mod = DATA.copy()
    G_xmin, G_xmax, G_ymin, G_ymax, G_zmin, G_zmax = assign_outer_shell_to_face_grains(
        DATA_mod, mask_outer, xmin, xmax, ymin, ymax, zmin, zmax
    )
    SURFACE_GRAINS = {G_xmin, G_xmax, G_ymin, G_ymax, G_zmin, G_zmax}
    logger.info("surface grain IDs: %s", SURFACE_GRAINS)

    surface_all, line_all, point_all, _ = classify_topology_fast(
        DATA_mod, k=k, min_other_atoms=min_other_atoms
    )

    if len(point_all) == 0:
        logger.warning("No quadruple points found; cannot build mesh")
        voxel_all = pd.DataFrame(
            columns=["atom-ID", "grain-ID", "voxel-X", "voxel-Y", "voxel-Z", "phase", "CI"]
        )
    else:
        quad_unique = build_quad_unique(point_all)
        segments_df = build_triple_line_segments(line_all, quad_unique)
        quad_pos = quad_unique[["X_mean", "Y_mean", "Z_mean"]].to_numpy()
        S_min = quad_pos.min(axis=0) * 0.95
        S_min_global = S_min.copy()

        pair_to_mesh = build_pair_to_mesh(surface_all, line_all, quad_unique, segments_df, voxel_size)
        pair_to_mesh_shift = {}
        for pair, patches in pair_to_mesh.items():
            new_patches = []
            for p in patches:
                new_patches.append(
                    {
                        "verts": np.asarray(p["verts"], dtype=float) - S_min_global,
                        "simplices": p["simplices"],
                        "i": p["i"],
                        "j": p["j"],
                        "k": p["k"],
                    }
                )
            pair_to_mesh_shift[pair] = new_patches

        df_atoms = DATA_mod.copy()
        df_atoms["grain-ID"] = df_atoms["grain-ID"].astype(int)
        real_grains = sorted(
            g for g in df_atoms["grain-ID"].unique().astype(int) if g not in SURFACE_GRAINS
        )
        grain_centers = {}
        for g in real_grains:
            pts = df_atoms.loc[df_atoms["grain-ID"] == g, ["X", "Y", "Z"]].to_numpy()
            if pts.shape[0] == 0:
                continue
            center_world = pts.mean(axis=0)
            center_shift = center_world - S_min_global
            grain_centers[g] = center_shift
        logger.info("[grain_centers] real grains=%d, with centers=%d", len(real_grains), len(grain_centers))

        idx_by_gid = {}
        centers_by_gid = {}
        for gid in real_grains:
            idx_in, centers_in = voxelize_grain_on_integer_grid(
                gid,
                pair_to_mesh_shift,
                grain_centers,
                voxel_size,
                S_min_global,
                index_step=1,
                margin_nvox=1,
            )
            if idx_in is None or len(idx_in) == 0:
                logger.debug(f"[skip] grain {gid} has no voxels")
                continue
            idx_by_gid[gid] = idx_in
            centers_by_gid[gid] = centers_in
            logger.info(f"[ok] grain {gid}: {len(idx_in)} voxels")

        if not idx_by_gid:
            raise RuntimeError("No voxels generated for any grain")
        all_idx = np.vstack(list(idx_by_gid.values()))
        global_min = all_idx.min(axis=0).astype(int)
        logger.info("全局 index 最小值 global_min = %s", global_min)

        rows = []
        atom_counter = 1
        for gid, idx_in in idx_by_gid.items():
            idx_norm = idx_in.astype(int) - global_min[None, :]
            voxel_xyz = idx_norm * int(voxel_size)
            n = idx_in.shape[0]
            phase = np.ones(n, dtype=float)
            CI = np.ones(n, dtype=float)
            atom_id = np.arange(atom_counter, atom_counter + n, dtype=int)
            atom_counter += n
            df_g = pd.DataFrame(
                {
                    "atom-ID": atom_id,
                    "grain-ID": np.full(n, int(gid), dtype=int),
                    "voxel-X": voxel_xyz[:, 0].astype(int),
                    "voxel-Y": voxel_xyz[:, 1].astype(int),
                    "voxel-Z": voxel_xyz[:, 2].astype(int),
                    "phase": phase,
                    "CI": CI,
                }
            )
            rows.append(df_g)
        voxel_all = pd.concat(rows, ignore_index=True)
        voxel_all.sort_values(
            by=["grain-ID", "voxel-X", "voxel-Y", "voxel-Z"],
            inplace=True,
        )
        logger.info("最终 voxel_all 大小: %s", voxel_all.shape)

        voxel_all = map_to_full_grid(voxel_all, voxel_size)

    # Normalize voxel coordinates to increment by 1 (divide by voxel_size)
    voxel_all["voxel-X"] = (voxel_all["voxel-X"] / voxel_size).astype(int)
    voxel_all["voxel-Y"] = (voxel_all["voxel-Y"] / voxel_size).astype(int)
    voxel_all["voxel-Z"] = (voxel_all["voxel-Z"] / voxel_size).astype(int)

    voxel_all.to_csv(output_path, sep=" ", index=False)
    logger.info("完整 0-based 长方体网格文件已保存到: %s", output_path)
    return output_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract GB topology from LAMMPS dump and generate voxel_all.csv with 0-based grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dump", type=Path, required=True, help="Path to LAMMPS dump file (one timestep)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., check_large_voxel_from_mesh_what.csv)",
    )
    parser.add_argument(
        "--cube-ratio",
        type=float,
        default=0.015,
        dest="crop_ratio",
        help="Crop ratio per side (default 0.015)",
    )
    parser.add_argument("--k", type=int, default=25, help="k-NN for classification (default 25)")
    parser.add_argument(
        "--min-other-atoms",
        type=int,
        default=4,
        help="Min other-grain neighbors (default 4)",
    )
    parser.add_argument("--voxel-size", type=float, default=2.0, help="Voxel size (default 2.0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(name)s: %(message)s",
        force=True,
    )
    run(
        dump_path=args.dump,
        output_path=args.output,
        crop_ratio=args.crop_ratio,
        k=args.k,
        min_other_atoms=args.min_other_atoms,
        voxel_size=args.voxel_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
