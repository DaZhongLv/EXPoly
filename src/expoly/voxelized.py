#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voxelize_dump.py  —  Project an atomic structure (LAMMPS dump, single timestep)
onto a regular 3D voxel grid using unconditional nearest-neighbor assignment.

Outputs two files next to the dump under D3D_project/:
  - Characterize_<dump_basename>.<ext>
  - Original_ID_<dump_basename>.<ext>

No cutoff is used: every voxel is always assigned to its nearest atom.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ------------------------------ Logging ---------------------------------------

logger = logging.getLogger("voxelize_dump")


# ------------------------------ Config ----------------------------------------

@dataclass
class VoxelizeConfig:
    dump_path: Path
    cube_ratio: float
    out_dir: Optional[Path] = None
    log_level: str = "INFO"

    # column mapping (override via CLI if your dump uses different names)
    col_id: str = "id"
    col_x: str = "x"
    col_y: str = "y"
    col_z: str = "z"
    col_qw: str = "quatw"
    col_qx: str = "quati"
    col_qy: str = "quatj"
    col_qz: str = "quatk"
    col_grain: str = "Grain"


# ------------------------------ I/O helpers -----------------------------------

def _read_dump_header(dump_path: Path) -> Tuple[int, int, Tuple[float, float], Tuple[float, float], Tuple[float, float], List[str]]:
    """
    Parse a single-timestep LAMMPS dump file to obtain:
      - atoms_header_line (0-based line index of 'ITEM: ATOMS ...')
      - n_atoms
      - (xlo,xhi), (ylo,yhi), (zlo,zhi)
      - atoms_columns (list of column names)
    Robust to extra spaces; targets the *last* occurrence if multiple timesteps exist.
    """
    atoms_header_line = -1
    atoms_columns: List[str] = []
    n_atoms = None
    xlo = xhi = ylo = yhi = zlo = zhi = None

    with dump_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("ITEM: NUMBER OF ATOMS"):
            if i + 1 < len(lines):
                try:
                    n_atoms = int(lines[i + 1].strip())
                except Exception:
                    pass
                i += 2
                continue

        if s.startswith("ITEM: BOX BOUNDS"):
            # Next 3 lines are bounds
            if i + 3 < len(lines):
                try:
                    xlo, xhi = map(float, lines[i + 1].split()[:2])
                    ylo, yhi = map(float, lines[i + 2].split()[:2])
                    zlo, zhi = map(float, lines[i + 3].split()[:2])
                except Exception:
                    pass
                i += 4
                continue

        if s.startswith("ITEM: ATOMS"):
            atoms_header_line = i
            atoms_columns = s.split()[2:]  # tokens after 'ITEM: ATOMS'
            # do not break; if file contains multiple timesteps, keep last
        i += 1

    if atoms_header_line < 0 or n_atoms is None or xlo is None:
        raise RuntimeError(f"Failed to parse dump header: {dump_path}")

    return atoms_header_line, n_atoms, (xlo, xhi), (ylo, yhi), (zlo, zhi), atoms_columns


def _load_atoms_table(dump_path: Path, header_line: int, n_atoms: int, cols: List[str]) -> pd.DataFrame:
    """
    Read the atoms table using pandas, skipping to the line after 'ITEM: ATOMS'.
    """
    df = pd.read_csv(
        dump_path,
        delim_whitespace=True,
        names=cols,
        skiprows=header_line + 1,
        nrows=n_atoms,
        engine="c",
    )
    # Normalize dtypes to numeric when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


# ------------------------------ Grid generation --------------------------------

def _ceil_to_multiple(x: float, step: float) -> float:
    return float(np.ceil(x / step) * step)


def _generate_voxel_grid(xlo: float, xhi: float,
                         ylo: float, yhi: float,
                         zlo: float, zhi: float,
                         cube_ratio: float) -> np.ndarray:
    """
    Build regular voxel grid covering [0, S_max] in each dimension, where:
      S_max = ceil((domain_max - domain_min) / cube_ratio) * cube_ratio
    Then shift grid to start at 0 (we later store voxel coords after subtracting S_min).
    """
    S_min = np.array([0.0, 0.0, 0.0], dtype=float)
    S_max = np.array([
        _ceil_to_multiple(xhi - xlo, cube_ratio),
        _ceil_to_multiple(yhi - ylo, cube_ratio),
        _ceil_to_multiple(zhi - zlo, cube_ratio),
    ], dtype=float)

    x_bins = np.linspace(S_min[0], S_max[0], int(1 + (S_max[0] - S_min[0]) / cube_ratio))
    y_bins = np.linspace(S_min[1], S_max[1], int(1 + (S_max[1] - S_min[1]) / cube_ratio))
    z_bins = np.linspace(S_min[2], S_max[2], int(1 + (S_max[2] - S_min[2]) / cube_ratio))

    X, Y = np.meshgrid(x_bins, y_bins, indexing="xy")
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    XYZ = np.hstack([np.tile(XY, (len(z_bins), 1)),
                     np.repeat(z_bins, len(XY))[:, None]])
    return XYZ  # shape: (Nx*Ny*Nz, 3)


# ------------------------------ Column helpers ---------------------------------

def _ensure_columns(df: pd.DataFrame, cfg: VoxelizeConfig) -> Dict[str, str]:
    """
    Build a mapping from required semantic names -> actual df columns.
    If some optional columns are missing (e.g., grain), they will be synthesized with zeros.
    """
    cols = set(df.columns)
    # Must-have positions
    for need in (cfg.col_id, cfg.col_x, cfg.col_y, cfg.col_z):
        if need not in cols:
            raise KeyError(f"Required column '{need}' not found in dump (columns={sorted(cols)})")

    # Optional: quaternions & grain
    missing: List[str] = []
    for opt in (cfg.col_qw, cfg.col_qx, cfg.col_qy, cfg.col_qz, cfg.col_grain):
        if opt not in cols:
            missing.append(opt)

    if missing:
        logger.warning("Missing optional columns: %s. They will be filled with zeros (or ids for grain if absent).",
                       ", ".join(missing))
        # Create zero-filled columns for missing optionals
        for opt in missing:
            if opt == cfg.col_grain:
                # Grain falls back to zeros (float)
                df[opt] = 0.0
            else:
                df[opt] = 0.0

    return {
        "id": cfg.col_id,
        "x": cfg.col_x,
        "y": cfg.col_y,
        "z": cfg.col_z,
        "qw": cfg.col_qw,
        "qx": cfg.col_qx,
        "qy": cfg.col_qy,
        "qz": cfg.col_qz,
        "grain": cfg.col_grain,
    }


# ------------------------------ Core projection --------------------------------

def project_voxels(df_atoms: pd.DataFrame,
                   colmap: Dict[str, str],
                   grid_xyz: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Unconditional nearest-neighbor assignment from voxel centers to atoms.
    Returns:
      - Characterize DataFrame with expected columns
      - Original_ID Series
    """
    # KD-tree over atomic positions
    atoms_xyz = df_atoms[[colmap["x"], colmap["y"], colmap["z"]]].to_numpy(dtype=float, copy=False)
    tree = cKDTree(atoms_xyz)

    # Unconditional nearest neighbor (no cutoff)
    # idx: for each voxel, index of nearest atom
    _, idx = tree.query(grid_xyz, k=1, workers=-1)

    # Select required per-voxel attributes from the matched atoms
    take_cols = [colmap["id"], colmap["qw"], colmap["qx"], colmap["qy"], colmap["qz"], colmap["grain"]]
    picked = df_atoms.iloc[idx][take_cols].reset_index(drop=True).to_numpy()

    # Compose output table
    phase_ci = np.ones((len(grid_xyz), 2), dtype=float)  # phase=1, CI=1 (or whatever constant you prefer)
    out = np.hstack([picked, grid_xyz, phase_ci])

    characterize = pd.DataFrame(out, columns=[
        'atom-ID', 'ptm-qw', 'ptm-qx', 'ptm-qy', 'ptm-qz',
        'grain-ID', 'voxel-X', 'voxel-Y', 'voxel-Z', 'phase', 'CI'
    ])
    original_id = characterize['atom-ID'].astype(int)

    return characterize, original_id


# ------------------------------ Naming & saving --------------------------------

def _make_output_paths(dump_path: Path, out_dir: Optional[Path]) -> Tuple[Path, Path]:
    """
    Build paths:
      <dump_dir or out_dir>/D3D_project/Characterize_<dumpname>.<ext>
      <dump_dir or out_dir>/D3D_project/Original_ID_<dumpname>.<ext>
    """
    base_dir = (out_dir or dump_path.parent)
    d3d_dir = base_dir / "D3D_project"
    d3d_dir.mkdir(parents=True, exist_ok=True)

    base = dump_path.name
    stem, _ext = Path(base).stem, Path(base).suffix
    if stem.startswith("dump"):
        char_name = base.replace("dump", "Characterize", 1)
        id_name = base.replace("dump", "Original_ID", 1)
    else:
        char_name = f"Characterize_{base}"
        id_name = f"Original_ID_{base}"

    return d3d_dir / char_name, d3d_dir / id_name


def save_outputs(characterize: pd.DataFrame, original_id: pd.Series, char_path: Path, id_path: Path) -> None:
    # Shift voxel coordinates to start at 0 (already generated from 0..Smax, so this is NOP)
    # Kept for compatibility with older scripts that subtract S_min.
    for c in ("voxel-X", "voxel-Y", "voxel-Z"):
        characterize[c] = characterize[c] - characterize[c].min()

    characterize.to_csv(char_path, sep="\t", index=False, lineterminator="\n")
    original_id.to_csv(id_path, index=False, header=False)
    logger.info("Saved: %s", char_path)
    logger.info("Saved: %s", id_path)


# ------------------------------ CLI -------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Voxelize an atomic structure (LAMMPS dump → regular voxel grid via NN projection). No cutoff.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dump", required=True, help="Path to LAMMPS dump (single timestep).")
    p.add_argument("--cube-ratio", required=True, type=float, help="Voxel pitch (same units as dump coordinates).")
    p.add_argument("--out-dir", default=None, help="Override output base directory (default: dump directory).")

    # Column mapping (only change if your dump uses different names)
    p.add_argument("--col-id", default="id", help="Atom id column name in dump.")
    p.add_argument("--col-x", default="x", help="X column name in dump.")
    p.add_argument("--col-y", default="y", help="Y column name in dump.")
    p.add_argument("--col-z", default="z", help="Z column name in dump.")
    p.add_argument("--col-qw", default="quatw", help="Quaternion w column name in dump.")
    p.add_argument("--col-qx", default="quati", help="Quaternion x column name in dump.")
    p.add_argument("--col-qy", default="quatj", help="Quaternion y column name in dump.")
    p.add_argument("--col-qz", default="quatk", help="Quaternion z column name in dump.")
    p.add_argument("--col-grain", default="Grain", help="Grain id column name in dump (if absent will be zeros).")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> int:
    ap = build_argparser()
    ns = ap.parse_args()

    cfg = VoxelizeConfig(
        dump_path=Path(ns.dump).expanduser().resolve(),
        cube_ratio=float(ns.cube_ratio),
        out_dir=(None if ns.out_dir is None else Path(ns.out_dir).expanduser().resolve()),
        log_level=str(ns.log_level).upper(),
        col_id=ns.col_id, col_x=ns.col_x, col_y=ns.col_y, col_z=ns.col_z,
        col_qw=ns.col_qw, col_qx=ns.col_qx, col_qy=ns.col_qy, col_qz=ns.col_qz,
        col_grain=ns.col_grain,
    )

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                        format="%(levelname)s | %(message)s")

    # 1) Parse header & atoms table
    hdr_line, n_atoms, (xlo, xhi), (ylo, yhi), (zlo, zhi), cols = _read_dump_header(cfg.dump_path)
    logger.info("Parsed dump: n_atoms=%d; box=[%.6g..%.6g, %.6g..%.6g, %.6g..%.6g]",
                n_atoms, xlo, xhi, ylo, yhi, zlo, zhi)
    atoms = _load_atoms_table(cfg.dump_path, hdr_line, n_atoms, cols)

    # 2) Column mapping / synthesize missing optionals
    colmap = _ensure_columns(atoms, cfg)

    # 3) Build voxel grid in [0, Smax]^3 with pitch=cube_ratio
    grid = _generate_voxel_grid(xlo, xhi, ylo, yhi, zlo, zhi, cfg.cube_ratio)
    logger.info("Generated voxel grid: %d points", len(grid))

    # 4) NN projection (no cutoff)
    characterize, original_id = project_voxels(atoms, colmap, grid)

    # 5) Save
    char_path, id_path = _make_output_paths(cfg.dump_path, cfg.out_dir)
    save_outputs(characterize, original_id, char_path, id_path)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

