# src/expoly/carve.py
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import spatial

from . import general_func
from .frames import Frame  # used to query Euler angles / neighborhood / extensions

logger = logging.getLogger(__name__)

__all__ = [
    "CarveConfig",
    "unit_vec_ratio",
    "sc_to_fcc",
    "sc_to_bcc",
    "sc_to_dia",
    "move_to_center",
    "prepare_carve",
    "prepare_carve_meta",
    "iter_sc_box_chunks",
    "carve_points",
    "carve_points_inverse_box",
    "carve_gb_keep_m1",
    "maybe_make_frame_readonly",
    "process_pre",
    "process",
    "process_extend",
]

# ================================ Config =====================================


@dataclass
class CarveConfig:
    """
    Configuration for carving / lattice construction over the WHOLE grain volume.
    Fields are kept compatible with existing CLI code.
    """

    lattice: str = "FCC"  # "FCC" | "BCC" | "DIA"
    ratio: float = 1.5  # lattice spacing (in H-units)

    # --- backward-compat / still used by neighbor selection ---
    ci_radius: float = math.sqrt(2.0)  # radius (in H-units) to keep lattice pts near any H voxel

    # randomization for the spherical SC seed grid (legacy ball path)
    random_center: bool = False  # deterministic coverage by default
    rng_seed: Optional[int] = None  # set to reproducible seed if random_center=True

    # for extended pipeline
    unit_extend_ratio: int = 3

    # chunked inverse-box generation: cells per chunk along longest axis (conservative default)
    chunk_cells: int = 32


# ============================== Small utilities ===============================


def unit_vec_ratio(out_xyz: np.ndarray, unit_offsets: np.ndarray, ratio: float) -> np.ndarray:
    """
    Cartesian sum between N×3 'out_xyz' and M×3 lattice 'unit_offsets', scaled by 'ratio'.
    Returns (N*M)×3.
    """
    out_xyz = np.asarray(out_xyz, dtype=float).reshape(-1, 3)
    unit_offsets = np.asarray(unit_offsets, dtype=float).reshape(-1, 3)
    rep = np.repeat(out_xyz, len(unit_offsets), axis=0)
    til = np.tile(unit_offsets, (len(out_xyz), 1)) * float(ratio)
    return rep + til


def sc_to_fcc(data: np.ndarray, ratio: float) -> np.ndarray:
    """Simple-cubic points -> FCC offsets (scaled)."""
    unit_fcc = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)
    return unit_vec_ratio(data, unit_fcc, ratio)


def sc_to_bcc(data: np.ndarray, ratio: float) -> np.ndarray:
    """Simple-cubic points -> BCC offsets (scaled)."""
    unit_bcc = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    return unit_vec_ratio(data, unit_bcc, ratio)


def sc_to_dia(data: np.ndarray, ratio: float) -> np.ndarray:
    """Simple-cubic points -> Diamond offsets (scaled)."""
    unit_dia = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        dtype=float,
    )
    return unit_vec_ratio(data, unit_dia, ratio)


def move_to_center(points: np.ndarray, center_xyz: Sequence[float]) -> np.ndarray:
    """Translate points to a new center."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    c = np.asarray(center_xyz, dtype=float)
    pts[:, 0] += c[0]
    pts[:, 1] += c[1]
    pts[:, 2] += c[2]
    return pts


# ============================ Core: prepare & carve ============================


def prepare_carve_meta(
    out_df: pd.DataFrame,
    frame: Frame,
    cfg: CarveConfig,
    euler_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rotation matrix R, grain center, and crystal-frame AABB for inverse-box carve.
    Forward transform: pts_h = pts_c @ R.T + center  =>  inverse: vox_c = (vox_h - center) @ R.
    Returns (R, center, mins_c, maxs_c) with pad = ci_radius + ratio applied to AABB.
    """
    if "ID" in out_df.columns:
        grain_id = int(out_df["ID"].iloc[0])
    elif "grain-ID" in out_df.columns:
        grain_id = int(out_df["grain-ID"].iloc[0])
    else:
        raise ValueError("out_df must contain 'ID' or 'grain-ID'.")

    if euler_override is not None:
        avg_eul = np.asarray(euler_override, dtype=float).reshape(3)
    else:
        avg_eul = frame.search_avg_Euler(grain_id)
    R = general_func.eul2rot_bunge(avg_eul)

    hx_min, hx_max = float(out_df["HX"].min()), float(out_df["HX"].max())
    hy_min, hy_max = float(out_df["HY"].min()), float(out_df["HY"].max())
    hz_min, hz_max = float(out_df["HZ"].min()), float(out_df["HZ"].max())
    center = np.array(
        [0.5 * (hx_max + hx_min), 0.5 * (hy_max + hy_min), 0.5 * (hz_max + hz_min)],
        dtype=float,
    )

    vox_h = out_df[["HX", "HY", "HZ"]].to_numpy(dtype=float)
    vox_c = (vox_h - center) @ R
    pad = float(cfg.ci_radius) + float(cfg.ratio)
    mins_c = vox_c.min(axis=0) - pad
    maxs_c = vox_c.max(axis=0) + pad
    return R, center, mins_c, maxs_c


def iter_sc_box_chunks(
    mins_c: np.ndarray,
    maxs_c: np.ndarray,
    step: float,
    chunk_cells: int,
    axis: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (mins_slab, maxs_slab) for each slab along the given axis.
    Boundaries are aligned to the step grid to avoid misalignment artifacts
    at chunk boundaries (irregular lattice spacing).
    """
    lo = float(mins_c[axis])
    hi = float(maxs_c[axis])
    n_steps = max(1, int(np.ceil((hi - lo) / step)))
    n_chunks = max(1, (n_steps + chunk_cells - 1) // chunk_cells)
    steps_per_chunk = max(1, (n_steps + n_chunks - 1) // n_chunks)

    for i in range(n_chunks):
        k_lo = i * steps_per_chunk
        k_hi = min((i + 1) * steps_per_chunk, n_steps)
        slab_lo = lo + k_lo * step
        slab_hi = lo + k_hi * step
        if slab_hi <= slab_lo:
            continue
        mins_slab = mins_c.copy()
        maxs_slab = maxs_c.copy()
        mins_slab[axis] = slab_lo
        maxs_slab[axis] = slab_hi
        yield mins_slab, maxs_slab


def _sc_grid_in_box(mins: np.ndarray, maxs: np.ndarray, step: float) -> np.ndarray:
    """Simple-cubic grid points inside box [mins, maxs] with spacing step. Returns Nx3."""
    axes = [np.arange(mins[i], maxs[i] + 1e-9, step, dtype=float) for i in range(3)]
    if any(len(ax) == 0 for ax in axes):
        return np.empty((0, 3), dtype=float)
    grid = np.stack(np.meshgrid(axes[0], axes[1], axes[2], indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def carve_points_inverse_box(
    out_df: pd.DataFrame,
    frame: Frame,
    cfg: CarveConfig,
    euler_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inverse-rotate + oriented box + chunked generation. Kept points in H space, Nx3.
    Uses tree_H.query(pts_h, k=1, distance_upper_bound=ci_radius) for filtering.
    """
    R, center, mins_c, maxs_c = prepare_carve_meta(
        out_df, frame, cfg, euler_override=euler_override
    )
    axis = int(np.argmax(maxs_c - mins_c))
    step = float(cfg.ratio)
    chunk_cells = int(getattr(cfg, "chunk_cells", 32))

    tree_H = spatial.cKDTree(out_df[["HX", "HY", "HZ"]].to_numpy(dtype=float))
    kept_chunks: list = []
    t0 = time.process_time()
    total_candidates = 0

    for mins_slab, maxs_slab in iter_sc_box_chunks(mins_c, maxs_c, step, chunk_cells, axis):
        sc_c = _sc_grid_in_box(mins_slab, maxs_slab, step)
        if len(sc_c) == 0:
            continue
        lattice_c = _choose_lattice(sc_c, cfg.lattice, cfg.ratio)
        pts_h = lattice_c @ R.T + center
        total_candidates += len(pts_h)
        dist, _ = tree_H.query(pts_h, k=1, distance_upper_bound=float(cfg.ci_radius))
        keep_mask = np.isfinite(dist)
        if np.any(keep_mask):
            kept_chunks.append(pts_h[keep_mask])

    kept = np.vstack(kept_chunks) if kept_chunks else np.empty((0, 3), dtype=float)

    # Deduplicate at chunk boundaries (overlap at slab_hi plane)
    if len(kept) > 0:
        tol = 1e-9
        kept_rounded = np.round(kept / tol) * tol
        _, unique_idx = np.unique(kept_rounded, axis=0, return_index=True)
        kept = kept[unique_idx]

    logger.info(
        "carve_points_inverse_box: kept %d/%d in %.3fs",
        len(kept),
        total_candidates,
        time.process_time() - t0,
    )
    return kept


def _ball_grid(
    radius: float, step: float, random_center: bool, rng: Optional[np.random.Generator]
) -> np.ndarray:
    """
    Build a simple-cubic grid within a sphere of 'radius' with spacing 'step'.
    If random_center=True, shift query center randomly in [0,2)^3 (keeps legacy feel).
    """
    one_dim = np.arange(-radius, radius + 1e-9, step, dtype=float)  # +eps to include edge
    cube_struct = np.array(
        [[i, j, k] for i in one_dim for j in one_dim for k in one_dim], dtype=float
    )

    tree = spatial.cKDTree(cube_struct)
    center = (
        (rng.random(3) * 2.0)
        if (random_center and rng is not None)
        else (np.array([0.0, 0.0, 0.0]))
    )
    idx = tree.query_ball_point(center, r=radius, workers=-1)
    return cube_struct[np.asarray(idx, dtype=int)]


def _choose_lattice(sc_points: np.ndarray, lattice: str, ratio: float) -> np.ndarray:
    lat = lattice.upper()
    if lat == "FCC":
        return sc_to_fcc(sc_points, ratio)
    if lat == "BCC":
        return sc_to_bcc(sc_points, ratio)
    if lat == "DIA":
        return sc_to_dia(sc_points, ratio)
    raise ValueError(f"Unknown lattice '{lattice}'. Choose from 'FCC'|'BCC'|'DIA'.")


def prepare_carve(
    out_df: pd.DataFrame,
    frame: Frame,
    cfg: CarveConfig,
    euler_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a ball-shaped lattice cloud covering the WHOLE grain volume:
      1) Use bounding box of the grain to estimate center and circumscribed radius.
      2) Build SC points inside that sphere (spacing = cfg.ratio).
      3) Map SC points to the requested lattice (FCC/BCC/DIA).
      4) Rotate by grain's average Euler (Bunge) and translate to the grain center.

    If euler_override is provided (shape (3,) Bunge Euler angles), it is used
    instead of the frame's orientation for this grain (e.g. for random-orientation mode).
    """
    if "ID" in out_df.columns:
        grain_id = int(out_df["ID"].iloc[0])
    elif "grain-ID" in out_df.columns:
        grain_id = int(out_df["grain-ID"].iloc[0])
    else:
        raise ValueError("out_df must contain 'ID' or 'grain-ID'.")

    if euler_override is not None:
        avg_eul = np.asarray(euler_override, dtype=float).reshape(3)
    else:
        avg_eul = frame.search_avg_Euler(grain_id)
    # Your general_func.eul2rot implements Bunge convention
    R = general_func.eul2rot_bunge(avg_eul)  # 3x3

    hx_min, hx_max = float(out_df["HX"].min()), float(out_df["HX"].max())
    hy_min, hy_max = float(out_df["HY"].min()), float(out_df["HY"].max())
    hz_min, hz_max = float(out_df["HZ"].min()), float(out_df["HZ"].max())
    center = [0.5 * (hx_max + hx_min), 0.5 * (hy_max + hy_min), 0.5 * (hz_max + hz_min)]

    # radius based on diagonal * 1.5 (keeps your legacy hyperparameter)
    crave_r = math.dist([hx_min, hy_min, hz_min], [hx_max, hy_max, hz_max]) * 0.5 * 1.5

    rng = (
        np.random.default_rng(cfg.rng_seed) if cfg.rng_seed is not None else np.random.default_rng()
    )
    t0 = time.process_time()
    sc_points = _ball_grid(crave_r, cfg.ratio, cfg.random_center, rng)
    logger.info(
        "prepare_carve: SC sphere in %.3fs (points=%d)", time.process_time() - t0, len(sc_points)
    )

    lattice_pts = _choose_lattice(sc_points, cfg.lattice, cfg.ratio)
    rotated = lattice_pts @ R.T
    translated = move_to_center(rotated, center)
    return translated


def carve_points(
    out_df: pd.DataFrame,
    frame: Frame,
    cfg: CarveConfig,
    euler_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Keep lattice points that lie within 'cfg.ci_radius' (in H space) of ANY voxel in 'out_df'.
    Uses inverse-rotate + oriented box + chunked generation; KDTree filter via k=1 + distance_upper_bound.
    If euler_override is provided (Bunge Euler (3,)), it is used for rotation instead of frame.
    """
    return carve_points_inverse_box(out_df, frame, cfg, euler_override=euler_override)


def maybe_make_frame_readonly(frame: Frame) -> None:
    """
    Set Frame's large numpy arrays to read-only so that under fork(), copy-on-write
    does not duplicate them when child processes only read. Shared are the loaded
    arrays, not a live h5py file handle.
    """
    for name in ("GrainId", "Euler", "Num_NN", "Num_list", "Dimension", "fid", "eul"):
        arr = getattr(frame, name, None)
        if arr is not None and isinstance(arr, np.ndarray):
            try:
                arr.setflags(write=False)
            except ValueError:
                pass


# ============================ Margin / GB filtering ============================


def carve_gb_keep_m1(margin_df: pd.DataFrame, lattice_points: np.ndarray) -> pd.DataFrame:
    """
    Keep lattice points whose rounded HXYZ fall on cells with margin-ID in {0, 2}.
    Returns DataFrame with columns ['X','Y','Z','HX','HY','HZ','margin-ID'].
    """
    if lattice_points.size == 0:
        return pd.DataFrame(columns=["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID"])

    def round_to_cell(arr):
        return np.rint(arr)

    margin_sub = margin_df[["HX", "HY", "HZ", "margin-ID"]].copy()

    rounded = np.vectorize(round_to_cell, signature="(n)->(m)")(lattice_points)
    concat = np.hstack((lattice_points, rounded))
    df = pd.DataFrame(concat, columns=["X", "Y", "Z", "HX", "HY", "HZ"])
    merged = df.merge(margin_sub, on=["HX", "HY", "HZ"], how="left")
    keep = merged[(merged["margin-ID"] == 2) | (merged["margin-ID"] == 0)].copy()
    return keep


# ============================== Grain pipelines ===============================


def process_pre(
    grain_id: int,
    frame: Frame,
    cfg: CarveConfig,
    grain_euler_override: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """
    Prepare carved lattice cloud for a single grain (no GB filtering).
    If grain_euler_override is provided, use it for this grain's orientation.
    """
    out_df = frame.from_ID_to_D(grain_id)
    if "ID" not in out_df.columns:
        out_df = out_df.copy()
        out_df["ID"] = grain_id
    euler_override = grain_euler_override.get(grain_id) if grain_euler_override else None
    return carve_points(out_df, frame, cfg, euler_override=euler_override)


def process(
    grain_id: int,
    frame: Frame,
    cfg: CarveConfig,
    grain_euler_override: Optional[Dict[int, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Full M1-style pipeline for one grain:
      - compute lattice cloud (carve_points)
      - compute margin from frame.find_grain_NN_with_out()
      - filter lattice by M1 (margin-ID in {0,2})
    Returns DataFrame with ['X','Y','Z','HX','HY','HZ','margin-ID','grain-ID'].
    If grain_euler_override is provided, use it for this grain's orientation.
    """
    margin = frame.find_grain_NN_with_out(grain_id)
    carved = process_pre(grain_id, frame, cfg, grain_euler_override=grain_euler_override)
    kept = carve_gb_keep_m1(margin, carved)
    kept["grain-ID"] = grain_id
    return kept


def process_extend(
    grain_id: int,
    frame: Frame,
    cfg: CarveConfig,
    grain_euler_override: Optional[Dict[int, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Extended pipeline using frame extensions before carving.
    Requires: Frame.get_extend_Out_() and Frame.renew_outer_margin().
    If grain_euler_override is provided, use it for this grain's orientation.
    """
    out_margin = frame.find_grain_NN_with_out(grain_id)
    extend = frame.get_extend_Out_(out_margin, cfg.unit_extend_ratio)
    extend = frame.renew_outer_margin(extend)
    extend_xyz = extend.rename(columns={"grain-ID": "ID"}).copy()

    euler_override = grain_euler_override.get(grain_id) if grain_euler_override else None
    carved = carve_points(extend_xyz, frame, cfg, euler_override=euler_override)
    kept = carve_gb_keep_m1(extend, carved)
    kept["grain-ID"] = grain_id
    return kept
