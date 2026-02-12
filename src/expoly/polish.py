# src/expoly/polish.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

__all__ = [
    "PolishConfig",
    "write_lammps_input_data",
    "ovito_delete_overlap_data",
    "build_final_data_from_ovito_atoms",
    "polish_pipeline",
]

PathLike = Union[str, Path]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class PolishConfig:
    """
    Config for the 'raw_points -> LAMMPS data' polishing pipeline (OVITO is mandatory).

    'scan_ratio' MUST be computed by the CLI from lattice_constant / cube_ratio:
        scan_ratio = lattice_constant / cube_ratio

    Ranges are H-space (HX/HY/HZ). After cropping, (X,Y,Z) are shifted to the
    [H_low,*] origin and multiplied by scan_ratio.
    """

    scan_ratio: float = 1.0
    cube_ratio: float = 1.5

    hx_range: Sequence[float] = (0.0, 0.0)
    hy_range: Sequence[float] = (0.0, 0.0)
    hz_range: Sequence[float] = (0.0, 0.0)

    # If True, multiply the H-ranges by unit_extend_ratio (legacy 'real_extent').
    real_extent: bool = False
    unit_extend_ratio: int = 3

    current_frame: int = 0

    # OVITO overlap cutoff (same unit as final coordinates).
    ovito_cutoff: float = 1.6
    # Optional seed to shuffle delete order (avoids order bias, reproducible if set).
    ovito_shuffle_seed: Optional[int] = None

    # Masses section in final LAMMPS data. Default Ni.
    atom_mass: float = 58.6934

    # Whether to keep tmp files (tmp_in / ovito_psc / mask).
    keep_tmp: bool = False

    # Overwrite existing outputs.
    overwrite: bool = True


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def _ensure_parent(path: PathLike, overwrite: bool = True) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"File exists and overwrite=False: {p}")
    return p


def _load_raw_points(raw_path: PathLike) -> pd.DataFrame:
    """
    Read merged carved points. Supports .parquet (preferred) or .csv (no header):
      columns: X, Y, Z, HX, HY, HZ, margin-ID, grain-ID
    Force numeric & drop invalid rows to avoid dtype issues.
    """
    path = Path(raw_path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        # Ensure expected column names and order
        expected = ["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID", "grain-ID"]
        for c in expected:
            if c not in df.columns:
                raise KeyError(f"Parquet missing column {c!r}; got {list(df.columns)}")
        df = df[expected]
    else:
        df = pd.read_csv(raw_path, header=None, sep=r"\s+", low_memory=False)
        df.columns = ["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID", "grain-ID"]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d invalid rows while loading raw points.", before - len(df))
    return df


def _resolved_ranges(
    cfg: PolishConfig,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    hx = np.array(cfg.hx_range, dtype=float)
    hy = np.array(cfg.hy_range, dtype=float)
    hz = np.array(cfg.hz_range, dtype=float)
    if cfg.real_extent:
        hx *= cfg.unit_extend_ratio
        hy *= cfg.unit_extend_ratio
        hz *= cfg.unit_extend_ratio
    return (float(hx[0]), float(hx[1])), (float(hy[0]), float(hy[1])), (float(hz[0]), float(hz[1]))


# -----------------------------------------------------------------------------
# 1) Write a minimal but complete LAMMPS "data" (pre-OVITO; header + Masses + Atoms)
# -----------------------------------------------------------------------------


def _render_lammps_header(
    atom_num: int,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    zlo: float,
    zhi: float,
    atom_mass: float,
) -> str:
    # Minimal header that OVITO can read and that LAMMPS accepts.
    return (
        f"# EXPoly polished (pre-OVITO)\n"
        f"{atom_num} atoms\n"
        f"1 atom types\n\n"
        f"{xlo} {xhi} xlo xhi\n"
        f"{ylo} {yhi} ylo yhi\n"
        f"{zlo} {zhi} zlo zhi\n\n"
        f"Masses\n\n"
        f"1 {atom_mass}\n\n"
        f"Atoms # atomic\n\n"
    )


def write_lammps_input_data(
    raw_csv: PathLike,
    cfg: PolishConfig,
    out_in_path: PathLike,
    out_id_path: Optional[PathLike] = None,
) -> Tuple[int, Tuple[float, float, float, float, float, float]]:
    """
    Build a *complete* LAMMPS data file (header + Masses + Atoms) from raw carved points for OVITO input.
    Returns (atom_num, box_bounds).
    """
    df = _load_raw_points(raw_csv)
    (HX_lo, HX_hi), (HY_lo, HY_hi), (HZ_lo, HZ_hi) = _resolved_ranges(cfg)

    cut = df[
        (df["HX"] >= HX_lo)
        & (df["HX"] <= HX_hi)
        & (df["HY"] >= HY_lo)
        & (df["HY"] <= HY_hi)
        & (df["HZ"] >= HZ_lo)
        & (df["HZ"] <= HZ_hi)
    ].copy()
    if cut.empty:
        raise RuntimeError(
            "write_lammps_input_data: No atoms after cropping. Check hx/hy/hz ranges."
        )

    # Shift to lower bound and scale
    cut["X"] = (cut["X"] - HX_lo) * cfg.scan_ratio
    cut["Y"] = (cut["Y"] - HY_lo) * cfg.scan_ratio
    cut["Z"] = (cut["Z"] - HZ_lo) * cfg.scan_ratio

    atom_num = int(cut.shape[0])
    xlo, xhi = float(cut["X"].min()), float(cut["X"].max())
    ylo, yhi = float(cut["Y"].min()), float(cut["Y"].max())
    zlo, zhi = float(cut["Z"].min()), float(cut["Z"].max())

    out_in_path = _ensure_parent(out_in_path, cfg.overwrite)

    # id/type
    cut = cut.reset_index(drop=True)
    cut["id"] = np.arange(1, atom_num + 1, dtype=int)
    cut["type"] = 1

    # header
    header = _render_lammps_header(atom_num, xlo, xhi, ylo, yhi, zlo, zhi, cfg.atom_mass)
    with open(out_in_path, "w", encoding="utf-8") as f:
        f.write(header)

    # Atoms
    cut[["id", "type", "X", "Y", "Z"]].to_csv(
        out_in_path, mode="a", sep=" ", index=False, header=False, float_format="%.10g"
    )
    logger.info("write_lammps_input_data: atoms=%d → %s", atom_num, out_in_path)

    # Optional ID map (.parquet or legacy space-sep CSV)
    if out_id_path is not None:
        out_id_path = _ensure_parent(out_id_path, True)
        id_df = cut[["id", "X", "Y", "Z", "margin-ID", "grain-ID"]]
        if str(out_id_path).endswith(".parquet"):
            id_df.to_parquet(out_id_path, index=False)
        else:
            id_df.to_csv(out_id_path, sep=" ", index=False, header=False)
        logger.info("write_lammps_input_data: id map → %s", out_id_path)

    return atom_num, (xlo, xhi, ylo, yhi, zlo, zhi)


# -----------------------------------------------------------------------------
# 2) OVITO de-duplication (data -> data), connected-components (one per cluster)
# -----------------------------------------------------------------------------


def ovito_delete_overlap_data(
    in_lammps_path: PathLike,
    out_overlap_mask_path: PathLike,
    out_psc_path: PathLike,
    cutoff: float,
    shuffle_seed: Optional[int] = None,
) -> None:
    """
    Delete near-duplicates using OVITO: build graph of pairs within cutoff via
    scipy cKDTree (no per-atom loop), connected components, one representative per
    component (first in shuffle order); OVITO DeleteSelectedModifier does the removal.

    Outputs:
      - out_overlap_mask_path: selection mask (0=keep, 1=delete) or .npy array
      - out_psc_path: cleaned LAMMPS data file
    """
    try:
        from ovito.io import export_file, import_file
        from ovito.modifiers import DeleteSelectedModifier
    except Exception as e:
        raise RuntimeError("ovito is required. Install it with `pip install ovito`.") from e

    in_lammps_path = Path(in_lammps_path)
    out_overlap_mask_path = _ensure_parent(out_overlap_mask_path, True)
    out_psc_path = _ensure_parent(out_psc_path, True)

    pipeline = import_file(str(in_lammps_path))

    def modifier(frame, data):
        N = data.particles.count
        # 1) Get positions (N, 3) for cKDTree
        positions = np.asarray(data.particles["Position"][...], dtype=np.float64)

        # 2) Build edge list with cKDTree (no for-loop over atoms)
        tree = cKDTree(positions)
        pairs = tree.query_pairs(float(cutoff))
        edges = np.array(list(pairs), dtype=np.intp) if pairs else np.zeros((0, 2), dtype=np.intp)

        # 3) Connected components (undirected)
        if edges.shape[0] > 0:
            row = np.concatenate([edges[:, 0], edges[:, 1]])
            col = np.concatenate([edges[:, 1], edges[:, 0]])
            adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(N, N))
            n_comp, labels = connected_components(adj, directed=False)
        else:
            n_comp, labels = N, np.arange(N, dtype=np.intp)

        # 4) One representative per component: first in shuffle order
        rng = np.random.default_rng(None if shuffle_seed is None else int(shuffle_seed))
        order = np.arange(N)
        rng.shuffle(order)
        first_in_order = np.full(n_comp, -1, dtype=np.intp)
        for idx in order:
            c = labels[idx]
            if first_in_order[c] == -1:
                first_in_order[c] = idx
        representatives = set(int(first_in_order[c]) for c in range(n_comp))

        # 5) Selection: 0 = keep, 1 = delete (vectorized)
        selection = np.ones(N, dtype=np.intp)
        selection[np.fromiter(representatives, dtype=np.intp)] = 0
        data.particles_.create_property("Selection", data=selection)

    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    mask = data.particles["Selection"][...]
    # Save as .npy when path has .npy extension, else CSV for backward compatibility
    if str(out_overlap_mask_path).endswith(".npy"):
        np.save(out_overlap_mask_path, mask, allow_pickle=False)
    else:
        np.savetxt(out_overlap_mask_path, mask, fmt="%d", delimiter=",")
    logger.info("ovito_delete_overlap_data: mask → %s", out_overlap_mask_path)

    pipeline.modifiers.append(DeleteSelectedModifier(operate_on={"particles"}))
    export_file(pipeline, str(out_psc_path), "lammps/data")
    logger.info("ovito_delete_overlap_data: cleaned data → %s", out_psc_path)


# -----------------------------------------------------------------------------
# 3) Extract Atoms from OVITO output and rebuild a fresh final.data
# -----------------------------------------------------------------------------


def _extract_atoms_lines_from_data(path: PathLike) -> List[str]:
    """
    Extract *data lines* from the 'Atoms' section of a LAMMPS data file (not including the 'Atoms...' title line).
    Assumes 'Atoms # atomic' style: lines have at least 5 fields: id type x y z.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    in_atoms = False
    for line in lines:
        s = line.strip()
        if not in_atoms:
            if s.startswith("Atoms"):
                in_atoms = True
            continue
        # in atoms block: stop when hitting a new section header (alpha), else collect numeric lines
        if s and s[0].isalpha():
            break
        if not s:
            continue
        out.append(line)
    if not out:
        raise RuntimeError(f"No atoms lines extracted from {path}")
    return out


def _parse_xyz_from_atoms_lines(atoms_lines: List[str]) -> np.ndarray:
    """
    Parse (x,y,z) from atoms lines; assumes 'id type x y z ...'.
    """
    vals = []
    for ln in atoms_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            vals.append((x, y, z))
        except Exception:
            pass
    if not vals:
        raise RuntimeError("Failed to parse any XYZ from atoms lines.")
    return np.asarray(vals, dtype=float)


def build_final_data_from_ovito_atoms(
    ovito_clean_path: PathLike,
    final_data_path: PathLike,
    atom_mass: float = 58.6934,
    grain_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, Tuple[float, float, float, float, float, float]]:
    """
    Use OVITO-cleaned *data* file, re-extract the Atoms block, recompute N/box, and write a fresh, minimal LAMMPS data:
      - Correct '{N} atoms'
      - Correct x/y/z bounds
      - 'Masses' with the specified atom mass
      - 'Atoms # atomic' with IDs renumbered 1..N (type/x/y/z kept)
      - Optionally append per-atom grain-ID as an extra column (if `grain_ids` is not None).
    """
    atoms_lines = _extract_atoms_lines_from_data(ovito_clean_path)
    types_list: List[int] = []
    xyz_list: List[Tuple[float, float, float]] = []
    for ln in atoms_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            types_list.append(int(parts[1]))
            xyz_list.append((float(parts[2]), float(parts[3]), float(parts[4])))
        except (ValueError, IndexError):
            continue
    atom_num = len(types_list)
    if atom_num == 0:
        raise RuntimeError("No atoms lines parsed from OVITO output.")
    types = np.array(types_list, dtype=np.intp)
    xyz = np.array(xyz_list, dtype=float)
    new_ids = np.arange(1, atom_num + 1, dtype=np.int64)

    # If grain_ids is provided, check length consistency; otherwise disable.
    if grain_ids is not None:
        try:
            grain_ids = np.asarray(list(grain_ids), dtype=int)
            if grain_ids.shape[0] != atom_num:
                logger.warning(
                    "build_final_data_from_ovito_atoms: grain_ids length (%d) "
                    "!= atom_num (%d); ignoring grain_ids.",
                    grain_ids.shape[0],
                    atom_num,
                )
                grain_ids = None
        except Exception as e:
            logger.warning(
                "build_final_data_from_ovito_atoms: failed to interpret grain_ids (%s); ignoring.",
                e,
            )
            grain_ids = None

    xlo, xhi = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    ylo, yhi = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    zlo, zhi = float(xyz[:, 2].min()), float(xyz[:, 2].max())

    # Header; if grain_ids is present, we keep Atoms style 'atomic' but add a comment line
    header = (
        f"# EXPoly polished (final)\n"
        f"{atom_num} atoms\n"
        f"1 atom types\n\n"
        f"{xlo} {xhi} xlo xhi\n"
        f"{ylo} {yhi} ylo yhi\n"
        f"{zlo} {zhi} zlo zhi\n\n"
        f"Masses\n\n"
        f"1 {atom_mass}\n\n"
        f"Atoms # atomic\n"
    )
    if grain_ids is not None:
        # Comment line describing the columns; harmless for both OVITO and LAMMPS
        header += "# id type x y z grain-ID\n\n"
    else:
        header += "\n"

    body = np.column_stack([new_ids, types, xyz])
    if grain_ids is not None:
        body = np.column_stack([body, grain_ids])

    final_data_path = _ensure_parent(final_data_path, True)
    with open(final_data_path, "w", encoding="utf-8") as f:
        f.write(header)
        if grain_ids is not None:
            np.savetxt(f, body, fmt="%d %d %.10g %.10g %.10g %d", delimiter=" ")
        else:
            np.savetxt(f, body, fmt="%d %d %.10g %.10g %.10g", delimiter=" ")

    logger.info(
        "build_final_data_from_ovito_atoms: final data → %s (atoms=%d, with_grain=%s)",
        final_data_path,
        atom_num,
        grain_ids is not None,
    )
    return atom_num, (xlo, xhi, ylo, yhi, zlo, zhi)


def build_final_dump_with_grain(
    ovito_clean_path: PathLike,
    final_dump_path: PathLike,
    grain_ids: Sequence[int],
    timestep: int = 0,
) -> Tuple[int, Tuple[float, float, float, float, float, float]]:
    """
    Write a LAMMPS dump file from OVITO-cleaned LAMMPS data (ovito_psc) and per-atom grain-ID:

        ITEM: TIMESTEP
        <timestep>
        ITEM: NUMBER OF ATOMS
        N
        ITEM: BOX BOUNDS pp pp pp
        xlo xhi
        ylo yhi
        zlo zhi
        ITEM: ATOMS id type x y z grain-ID
        ...

    - Renumbers IDs to 1..N (consistent with build_final_data_from_ovito_atoms).
    - type / x / y / z come from OVITO's Atoms block output.
    """
    atoms_lines = _extract_atoms_lines_from_data(ovito_clean_path)
    atom_num = len(atoms_lines)

    # --- Process grain_ids ---
    grain_ids = np.asarray(list(grain_ids), dtype=int)
    if grain_ids.shape[0] != atom_num:
        raise RuntimeError(
            f"build_final_dump_with_grain: grain_ids length ({grain_ids.shape[0]}) "
            f"!= atom_num ({atom_num})"
        )

    # --- Parse type and xyz ---
    types = []
    xyz = []
    for ln in atoms_lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        t = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        types.append(t)
        xyz.append((x, y, z))

    if len(xyz) != atom_num:
        raise RuntimeError(
            "build_final_dump_with_grain: parsed atom count mismatch: " f"{len(xyz)} vs {atom_num}"
        )

    xyz = np.asarray(xyz, dtype=float)

    xlo, xhi = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    ylo, yhi = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    zlo, zhi = float(xyz[:, 2].min()), float(xyz[:, 2].max())

    final_dump_path = _ensure_parent(final_dump_path, True)
    with open(final_dump_path, "w", encoding="utf-8") as f:
        # ---- header ----
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{int(timestep)}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{atom_num}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{xlo:.10g} {xhi:.10g}\n")
        f.write(f"{ylo:.10g} {yhi:.10g}\n")
        f.write(f"{zlo:.10g} {zhi:.10g}\n")
        f.write("ITEM: ATOMS id type x y z grain-ID\n")

        # ---- Data rows: renumber id = 1..N ----
        new_id = 1
        for t, (x, y, z), gid in zip(types, xyz, grain_ids):
            f.write(f"{new_id:d} {int(t):d} " f"{x:.10g} {y:.10g} {z:.10g} {int(gid):d}\n")
            new_id += 1

    logger.info(
        "build_final_dump_with_grain: final dump → %s (atoms=%d)", final_dump_path, atom_num
    )
    return atom_num, (xlo, xhi, ylo, yhi, zlo, zhi)


# -----------------------------------------------------------------------------
# 4) Resolve path keys (compatible with your CLI)
# -----------------------------------------------------------------------------


def _resolve_pipeline_paths(paths: dict) -> Tuple[Path, Path, Path, Path]:
    """
    Compatible keys:
      tmp_in        -> pre-OVITO data file (complete)
      ovito_mask    -> selection mask
      ovito_psc     -> OVITO-cleaned data
      final_lmp     -> final data

    Also accepts aliases like tmp_dump/ovito_dump/final_dump, etc.
    """

    def pick(*names: str) -> Path:
        for n in names:
            if n in paths and paths[n]:
                return Path(paths[n])
        raise KeyError(f"Missing any of keys: {names!r} in paths={list(paths.keys())}")

    tmp_in = pick("tmp_in", "tmp_data", "tmp_dump", "tmp")
    ovito_mask = pick("ovito_mask", "mask", "overlap_mask")
    ovito_psc = pick("ovito_psc", "ovito_data", "ovito_dump", "ovito_clean")
    final_lmp = pick("final_lmp", "final_data", "final_dump", "final")
    return tmp_in, ovito_mask, ovito_psc, final_lmp


# -----------------------------------------------------------------------------
# 5) Orchestrator (OVITO is mandatory). Produces a *complete* final.data
# -----------------------------------------------------------------------------


def polish_pipeline(
    raw_csv: PathLike,
    cfg: PolishConfig,
    paths: dict,
    final_with_grain: bool = False,
) -> Path:
    """
    raw_points.csv  → write tmp_polish.in.data (complete data, pre-OVITO)
                    → OVITO de-dup to ovito_psc.data (mandatory)
                    → build final.data from OVITO's Atoms block (correct N/box)
                    → optionally append per-atom grain-ID column in final.data
                    → optionally remove tmp files

    Returns the path to final.data.
    """
    tmp_in, ovito_mask, ovito_psc, final_out = _resolve_pipeline_paths(paths)

    # 1) Pre-OVITO data
    #    Also write id mapping: id X Y Z margin-ID grain-ID (.parquet or legacy .ids.txt)
    id_map_path = paths.get("id_map") or (ovito_mask.parent / "ids.parquet")
    atom_num0, box0 = write_lammps_input_data(
        raw_csv,
        cfg,
        out_in_path=tmp_in,
        out_id_path=id_map_path,
    )
    logger.info("polish: pre-OVITO atoms=%d, box=%s", atom_num0, np.round(box0, 6))

    # 2) OVITO (mandatory)
    ovito_delete_overlap_data(
        tmp_in,
        ovito_mask,
        ovito_psc,
        cutoff=cfg.ovito_cutoff,
        shuffle_seed=cfg.ovito_shuffle_seed,
    )

    # 2.5) If needed, compute grain-ID for surviving atoms
    grain_ids_final = None
    if final_with_grain:
        try:
            # Read id map: id X Y Z margin-ID grain-ID (.parquet or legacy CSV)
            if str(id_map_path).endswith(".parquet"):
                id_df = pd.read_parquet(id_map_path)
            else:
                id_df = pd.read_csv(
                    id_map_path,
                    sep=r"\s+",
                    header=None,
                    names=["id", "X", "Y", "Z", "margin-ID", "grain-ID"],
                )
            # Read mask: 0=keep, 1=delete (.npy or legacy CSV)
            if str(ovito_mask).endswith(".npy"):
                mask_arr = np.load(ovito_mask, allow_pickle=False)
            else:
                mask_arr = np.loadtxt(ovito_mask, dtype=int, delimiter=",")
            if mask_arr.ndim != 1:
                mask_arr = mask_arr.reshape(-1)

            if len(mask_arr) != len(id_df):
                logger.warning(
                    "polish: mask length (%d) != id map length (%d); "
                    "cannot safely propagate grain-ID to final.data.",
                    len(mask_arr),
                    len(id_df),
                )
            else:
                keep_idx = np.where(mask_arr == 0)[0]
                # Assume OVITO maintains original order after deletion, just removes selected rows:
                grain_ids_final = id_df.loc[keep_idx, "grain-ID"].to_numpy(dtype=int)
                logger.info(
                    "polish: grain-ID propagated for %d surviving atoms.",
                    grain_ids_final.shape[0],
                )
        except Exception as e:
            logger.warning(
                "polish: failed to build grain-ID mapping for final.data: %s "
                "(final will be written without grain-ID column).",
                e,
            )
            grain_ids_final = None

    # 3) Final data from OVITO output
    atom_num1, box1 = build_final_data_from_ovito_atoms(
        ovito_psc,
        final_out,
        atom_mass=cfg.atom_mass,
        grain_ids=None,
    )
    logger.info("polish: final atoms=%d, box=%s", atom_num1, np.round(box1, 6))

    # 3.5) If grain-ID is needed and mapping succeeded, also write a dump format
    if final_with_grain and (grain_ids_final is not None):
        final_dump = final_out.with_suffix(".dump")
        try:
            atom_num2, box2 = build_final_dump_with_grain(
                ovito_psc,
                final_dump,
                grain_ids_final,
                timestep=cfg.current_frame,  # e.g., use current_frame as timestep
            )
            logger.info(
                "polish: also wrote dump with grain-ID → %s (atoms=%d)",
                final_dump,
                atom_num2,
            )
        except Exception as e:
            logger.warning(
                "polish: failed to write dump-with-grainID (%s); "
                "final.data was written successfully.",
                e,
            )

    # 4) Cleanup
    if not cfg.keep_tmp:
        for p in [tmp_in, ovito_psc, ovito_mask, id_map_path]:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("polish: cleaned tmp files.")

    return final_out
