# src/expoly/cli.py
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from expoly.carve import (
    CarveConfig,
    maybe_make_frame_readonly,
    process,
    process_extend,
)
from expoly.frames import Frame, VoxelCSVFrame
from expoly.generate_voronoi import run as voronoi_run
from expoly.polish import PolishConfig, polish_pipeline

LOG = logging.getLogger("expoly.cli")


def get_mp_context(mp_start: str):
    """
    Return (context, use_fork_preload).
    - auto: on Linux use fork (workers inherit memory, no HDF5 reload); else spawn.
    - fork: use fork (Linux/SLURM: single Frame load in main, shared by workers).
    - spawn: use spawn (each worker loads Frame via initializer).
    When use_fork_preload is True, main must load frame once and set _worker_frame_cache
    before creating the Pool; do not use initializer.
    """
    if mp_start == "spawn":
        return mp.get_context("spawn"), False
    if mp_start == "fork":
        return mp.get_context("fork"), True
    if mp_start == "auto":
        if sys.platform.startswith("linux"):
            return mp.get_context("fork"), True
        return mp.get_context("spawn"), False
    raise ValueError(f"mp_start must be 'auto'|'fork'|'spawn', got {mp_start!r}")


# --------------------------- small helpers ---------------------------


def _parse_lattice_constant(s: str) -> Dict[str, float]:
    """
    Parse lattice constant spec. Returns dict mapping lattice type -> value (Å).
    Formats:
      - "FCC:3.524,BCC:2.87"  -> {"FCC": 3.524, "BCC": 2.87}
      - "3.524"               -> {"FCC": 3.524} (single value, backward compat)
    """
    s = s.strip()
    if not s:
        raise argparse.ArgumentTypeError("lattice-constant cannot be empty")
    result: Dict[str, float] = {}
    if ":" in s:
        for part in s.split(","):
            part = part.strip()
            if ":" not in part:
                raise argparse.ArgumentTypeError(
                    f"Multi-phase format requires LATTICE:value (e.g. FCC:3.524,BCC:2.87), got {part!r}"
                )
            lat, val = part.split(":", 1)
            lat = lat.strip().upper()
            if not lat:
                raise argparse.ArgumentTypeError(f"Empty lattice type in {part!r}")
            try:
                v = float(val.strip())
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid lattice constant value {val!r}")
            if lat in result:
                raise argparse.ArgumentTypeError(f"Duplicate lattice type {lat}")
            result[lat] = v
    else:
        try:
            v = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Single value must be a number (e.g. 3.524), or use LATTICE:value format (e.g. FCC:3.524,BCC:2.87)"
            )
        result["FCC"] = v  # backward compat: single value -> FCC
    return result


def _parse_range(s: str) -> Tuple[int, int]:
    """
    Parse 'a:b' (also tolerates '[a:b]', spaces, etc.) → (a, b), both int.
    """
    try:
        s = s.strip().lstrip("[").rstrip("]").strip()
        if ":" not in s:
            raise argparse.ArgumentTypeError(f"Range must be like '0:50', got {s!r}")
        a, b = s.split(":", 1)
        result = (int(a.strip()), int(b.strip()))
        return result
    except Exception:
        raise


def _mk_run_dir(root: Path | None = None) -> Path:
    ts = int(time.time())
    base = Path("runs") if root is None else Path(root)
    path = base / f"expoly-{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _voxel_csv_h_ranges(csv_path: Path) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Read a voxel CSV (e.g. voronoi.csv) and return H ranges that cover the full grid:
    (0, max_x), (0, max_y), (0, max_z) so that VoxelCSVFrame indexing does not go out of bounds.
    """
    df = pd.read_csv(csv_path, sep=r"\s+", comment="#", engine="python")
    for col in ["voxel-X", "voxel-Y", "voxel-Z"]:
        if col not in df.columns:
            raise KeyError(f"Voxel CSV missing column {col!r}: {csv_path}")
    max_x = int(df["voxel-X"].max())
    max_y = int(df["voxel-Y"].max())
    max_z = int(df["voxel-Z"].max())
    return (0, max_x), (0, max_y), (0, max_z)


def _init_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s: %(message)s",
        force=True,  # Override any existing configuration
        stream=sys.stdout,  # Ensure output goes to stdout
    )
    # Ensure output is flushed immediately (important for SLURM/supercomputers)
    # This ensures progress messages appear in SLURM output files in real-time
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Fallback if reconfigure not available


# --------------------------- grain selection ---------------------------


def _validate_lattice_constants(
    frame: Frame,
    gids: np.ndarray,
    lattice_constants: Dict[str, float],
    default_lattice: str,
) -> None:
    """
    Ensure lattice_constants covers all lattice types used by selected grains.
    Raises RuntimeError with details if any lattice type is missing.
    """
    if not hasattr(frame, "get_lattice_for_grain"):
        return
    lattice_to_grains: Dict[str, List[int]] = {}
    for gid in gids:
        gid = int(gid)
        lat = frame.get_lattice_for_grain(gid)
        if lat is None:
            lat = default_lattice
        lattice_to_grains.setdefault(lat, []).append(gid)
    missing = [lat for lat in lattice_to_grains if lat not in lattice_constants]
    if missing:
        details = []
        for lat in missing:
            gs = lattice_to_grains[lat]
            sample = gs[:5]
            sample_str = ", ".join(str(g) for g in sample)
            if len(gs) > 5:
                sample_str += f", ... ({len(gs)} total)"
            details.append(f"  - {lat} (grain IDs: {sample_str})")
        raise RuntimeError(
            "Lattice constant not specified for phase(s) used in the selected region:\n"
            + "\n".join(details)
            + "\n\nPlease provide --lattice-constant with all lattice types, e.g.:\n"
            f"  --lattice-constant {','.join(f'{lat}:<value>' for lat in sorted(set(lattice_to_grains.keys())))}"
        )


def _pick_grain_ids(
    f: Frame, hx: Tuple[int, int], hy: Tuple[int, int], hz: Tuple[int, int]
) -> np.ndarray:
    """
    Select grains (>0) within Dream3D voxel ranges.
    """
    gids = f.find_volume_grain_ID(hx, hy, hz, return_count=False)
    gids = np.asarray(gids, dtype=int)
    gids = gids[gids > 0]
    if gids.size == 0:
        raise RuntimeError(
            f"No positive grain id found within the provided H ranges "
            f"(HX={hx}, HY={hy}, HZ={hz}). "
            f"Volume dimensions: HX=[0,{f.HX_lim}), HY=[0,{f.HY_lim}), HZ=[0,{f.HZ_lim}). "
            f"Check that your H ranges are within these bounds and contain grains (ID > 0)."
        )
    return np.unique(gids)


# --------------------------- frame builder ---------------------------


def _build_frame_for_carve(
    dream3d_path: Path | str,
    voxel_csv: Path | None,
    h5_grain_dset: str | None,
    h5_euler_dset: str | None = None,
    h5_numneighbors_dset: str | None = None,
    h5_neighborlist_dset: str | None = None,
    h5_dimensions_dset: str | None = None,
    h5_phases_dset: str | None = None,
    h5_phase_name_dset: str | None = None,
) -> Frame:
    """
    Build Frame or VoxelCSVFrame based on whether voxel_csv is provided.
    All h5_*_dset parameters allow customizing HDF5 dataset names.
    """
    dream3d_path = Path(dream3d_path)

    if not dream3d_path.exists():
        raise FileNotFoundError(
            f"Dream3D file not found: {dream3d_path}. "
            f"Please check the file path and ensure the file exists."
        )

    # Build mapping with custom dataset names or defaults
    mapping = {
        "GrainId": h5_grain_dset or "FeatureIds",
        "Euler": h5_euler_dset or "EulerAngles",
        "Num_NN": h5_numneighbors_dset or "NumNeighbors",
        "Num_list": h5_neighborlist_dset or "NeighborList",
        "Dimension": h5_dimensions_dset or "DIMENSIONS",
    }

    # Map attribute names to CLI argument names for error messages
    attr_to_arg = {
        "GrainId": "--h5-grain-dset",
        "Euler": "--h5-euler-dset",
        "Num_NN": "--h5-numneighbors-dset",
        "Num_list": "--h5-neighborlist-dset",
        "Dimension": "--h5-dimensions-dset",
    }

    try:
        import inspect

        if voxel_csv is None:
            # Pure Dream3D path with customizable dataset names
            # Pass phase args only if Frame supports them (backward compat)
            sig = inspect.signature(Frame)
            if "h5_phases_dset" in sig.parameters:
                return Frame(
                    str(dream3d_path),
                    mapping=mapping,
                    h5_phases_dset=h5_phases_dset,
                    h5_phase_name_dset=h5_phase_name_dset,
                )
            return Frame(str(dream3d_path), mapping=mapping)
        else:
            if not Path(voxel_csv).exists():
                raise FileNotFoundError(
                    f"Voxel CSV file not found: {voxel_csv}. Please check the file path."
                )
            # Voxel-CSV + HDF5 combination
            sig = inspect.signature(VoxelCSVFrame)
            if "h5_phases_dset" in sig.parameters:
                return VoxelCSVFrame(
                    path=str(dream3d_path),
                    voxel_csv=str(voxel_csv),
                    h5_grain_dset=mapping["GrainId"],
                    mapping=mapping,
                    h5_phases_dset=h5_phases_dset,
                    h5_phase_name_dset=h5_phase_name_dset,
                )
            return VoxelCSVFrame(
                path=str(dream3d_path),
                voxel_csv=str(voxel_csv),
                h5_grain_dset=mapping["GrainId"],
                mapping=mapping,
            )
    except KeyError as e:
        dataset_name = str(e).strip("'\"")
        # Find which attribute/dataset failed by checking the error message
        # KeyError from find_dataset_keys contains the dataset name
        failed_attr = None
        for attr, ds_name in mapping.items():
            if ds_name == dataset_name:
                failed_attr = attr
                break

        # Get the appropriate CLI argument name
        arg_name = attr_to_arg.get(failed_attr, "--h5-*-dset") if failed_attr else "--h5-*-dset"

        raise RuntimeError(
            f"Missing dataset '{dataset_name}' in HDF5 file '{dream3d_path}'. "
            f"Expected path: DataContainers/*/CellData/{dataset_name} or similar. "
            f"If your dataset has a different name, use {arg_name} to specify it. "
            f"Use 'h5dump -n {dream3d_path}' to inspect the HDF5 structure."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Dream3D file '{dream3d_path}': {e}. "
            f"Please check that the file is a valid Dream3D HDF5 file."
        ) from e


# --------------------------- carve runner ---------------------------


# Global variable for worker process to cache Frame (loaded once per worker)
_worker_frame_cache: Frame | None = None
_worker_frame_args: dict | None = None


def _init_worker_frame(
    dream3d_path: str,
    voxel_csv: str | None,
    h5_grain_dset: str | None,
    h5_euler_dset: str | None,
    h5_numneighbors_dset: str | None,
    h5_neighborlist_dset: str | None,
    h5_dimensions_dset: str | None,
    h5_phases_dset: str | None = None,
    h5_phase_name_dset: str | None = None,
) -> None:
    """
    Initialize worker process: load Frame once and cache it.
    This avoids reloading HDF5 file for every task in the same worker.
    """
    global _worker_frame_cache, _worker_frame_args
    args_key = (
        dream3d_path,
        voxel_csv,
        h5_grain_dset,
        h5_euler_dset,
        h5_numneighbors_dset,
        h5_neighborlist_dset,
        h5_dimensions_dset,
        h5_phases_dset,
        h5_phase_name_dset,
    )
    # Only reload if arguments changed (shouldn't happen, but safety check)
    if _worker_frame_args != args_key:
        _worker_frame_cache = _build_frame_for_carve(
            Path(dream3d_path),
            voxel_csv=Path(voxel_csv) if voxel_csv is not None else None,
            h5_grain_dset=h5_grain_dset,
            h5_euler_dset=h5_euler_dset,
            h5_numneighbors_dset=h5_numneighbors_dset,
            h5_neighborlist_dset=h5_neighborlist_dset,
            h5_dimensions_dset=h5_dimensions_dset,
            h5_phases_dset=h5_phases_dset,
            h5_phase_name_dset=h5_phase_name_dset,
        )
        _worker_frame_args = args_key


def _carve_one(args) -> pd.DataFrame:
    """
    Subprocess worker: select process based on extend flag; raises exception on failure (main process logs).
    Uses cached Frame from _init_worker_frame to avoid reloading HDF5 for each task.
    """
    global _worker_frame_cache

    (
        grain_id,
        dream3d_path,
        hx,
        hy,
        hz,
        lattice,
        ratio,
        extend,
        unit_extend_ratio,
        seed,
        voxel_csv,
        h5_grain_dset,
        h5_euler_dset,
        h5_numneighbors_dset,
        h5_neighborlist_dset,
        h5_dimensions_dset,
        grain_euler_override,
        h5_phases_dset,
        h5_phase_name_dset,
        effective_ratio,
    ) = args

    # Use cached Frame (loaded once per worker via initializer)
    if _worker_frame_cache is None:
        # Fallback: load if cache not initialized (shouldn't happen with proper initializer)
        _worker_frame_cache = _build_frame_for_carve(
            Path(dream3d_path),
            voxel_csv=Path(voxel_csv) if voxel_csv is not None else None,
            h5_grain_dset=h5_grain_dset,
            h5_euler_dset=h5_euler_dset,
            h5_numneighbors_dset=h5_numneighbors_dset,
            h5_neighborlist_dset=h5_neighborlist_dset,
            h5_dimensions_dset=h5_dimensions_dset,
            h5_phases_dset=h5_phases_dset,
            h5_phase_name_dset=h5_phase_name_dset,
        )
    frame = _worker_frame_cache

    # Per-grain lattice from phase (Phases/PhaseName) if available; else use global --lattice
    lattice_use = (
        frame.get_lattice_for_grain(grain_id) if hasattr(frame, "get_lattice_for_grain") else None
    )
    lattice_use = lattice_use or lattice

    # Per-phase ratio for unified physical scale (effective_ratio from min lattice constant)
    ratio_use = float(effective_ratio.get(lattice_use, ratio)) if effective_ratio else float(ratio)

    ccfg = CarveConfig(
        lattice=lattice_use,
        ratio=ratio_use,
        unit_extend_ratio=int(unit_extend_ratio),
        rng_seed=None if seed is None else int(seed),
    )

    if extend:
        df = process_extend(grain_id, frame, ccfg, grain_euler_override=grain_euler_override)
    else:
        df = process(grain_id, frame, ccfg, grain_euler_override=grain_euler_override)

    # Required columns; add lattice for multi-phase polish
    cols = ["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID", "grain-ID"]
    df = df[cols].copy()
    df["lattice"] = lattice_use
    return df


def _carve_all(
    dream3d: Path,
    hx: Tuple[int, int],
    hy: Tuple[int, int],
    hz: Tuple[int, int],
    lattice: str,
    ratio: float,
    extend: bool,
    unit_extend_ratio: int,
    workers: int,
    seed: int | None,
    voxel_csv: Path | None,
    h5_grain_dset: str | None,
    h5_euler_dset: str | None = None,
    h5_numneighbors_dset: str | None = None,
    h5_neighborlist_dset: str | None = None,
    h5_dimensions_dset: str | None = None,
    h5_phases_dset: str | None = None,
    h5_phase_name_dset: str | None = None,
    lattice_constants: Dict[str, float] | None = None,
    random_orientation: bool = False,
    run_dir: Path | None = None,
    mp_start: str = "auto",
) -> pd.DataFrame:
    # Load Frame once in main process. Under fork (Linux/SLURM) workers inherit this
    # copy; we make arrays read-only to avoid copy-on-write duplication. Shared are
    # the loaded numpy arrays, not a live h5py file handle.
    frame = _build_frame_for_carve(
        dream3d,
        voxel_csv=voxel_csv,
        h5_grain_dset=h5_grain_dset,
        h5_euler_dset=h5_euler_dset,
        h5_numneighbors_dset=h5_numneighbors_dset,
        h5_neighborlist_dset=h5_neighborlist_dset,
        h5_dimensions_dset=h5_dimensions_dset,
        h5_phases_dset=h5_phases_dset,
        h5_phase_name_dset=h5_phase_name_dset,
    )
    ctx, use_fork_preload = get_mp_context(mp_start)
    if use_fork_preload:
        maybe_make_frame_readonly(frame)
    voxel_csv_str = str(voxel_csv) if voxel_csv is not None else None
    _init_args = (
        str(dream3d),
        voxel_csv_str,
        h5_grain_dset,
        h5_euler_dset,
        h5_numneighbors_dset,
        h5_neighborlist_dset,
        h5_dimensions_dset,
        h5_phases_dset,
        h5_phase_name_dset,
    )
    global _worker_frame_cache, _worker_frame_args
    _worker_frame_cache = frame
    _worker_frame_args = _init_args

    gids = _pick_grain_ids(frame, hx, hy, hz)
    total_grains = len(gids)
    LOG.info("carve: %d grains selected in H ranges HX=%s HY=%s HZ=%s", total_grains, hx, hy, hz)

    if lattice_constants:
        _validate_lattice_constants(frame, gids, lattice_constants, lattice)

    # Per-phase effective ratio: use min lattice constant as reference so all phases
    # have the same physical extent per voxel (unified grain length scale)
    effective_ratio: Dict[str, float] = {}
    if lattice_constants and len(lattice_constants) > 0:
        ref_lat = min(lattice_constants.values())
        effective_ratio = {
            lat: lat_val * (ratio / ref_lat) for lat, lat_val in lattice_constants.items()
        }
        LOG.info(
            "[multi-phase] ref_lat=%.4f (min), effective_ratio=%s",
            ref_lat,
            effective_ratio,
        )

    # Build grain→Euler override map for random-orientation mode (shuffle only among grains in H range)
    grain_euler_override: Dict[int, np.ndarray] | None = None
    if random_orientation:
        LOG.info(
            "[random-orientation] Shuffling orientations only among %d grains in H range (hx=%s hy=%s hz=%s)...",
            total_grains,
            hx,
            hy,
            hz,
        )
        # Original list: grain IDs in the selected H range only
        original_list = [int(g) for g in gids]
        shuffled_list = original_list.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled_list)

        mask_selected = np.isin(frame.fid, original_list)
        orientation_mapping_pairs: List[
            Tuple[int, int]
        ] = []  # (grain_id, orientation_from_grain_id)
        if np.any(mask_selected):
            df_euler = pd.DataFrame(
                {
                    "grain_id": frame.fid[mask_selected],
                    "e1": frame.eul[mask_selected, 0],
                    "e2": frame.eul[mask_selected, 1],
                    "e3": frame.eul[mask_selected, 2],
                }
            )
            grain_means = df_euler.groupby("grain_id")[["e1", "e2", "e3"]].mean()

            grain_euler_override = {}
            for grain_id in original_list:
                shuffled_index = shuffled_list.index(grain_id)
                euler_source_grain_id = original_list[shuffled_index]
                orientation_mapping_pairs.append((grain_id, euler_source_grain_id))
                if euler_source_grain_id in grain_means.index:
                    euler = grain_means.loc[euler_source_grain_id].values
                    grain_euler_override[grain_id] = euler.copy()
                else:
                    LOG.warning(
                        "[random-orientation] Grain %d (mapped from %d) not found, using zero orientation",
                        grain_id,
                        euler_source_grain_id,
                    )
                    grain_euler_override[grain_id] = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            LOG.warning("[random-orientation] No selected grains found in data")
            grain_euler_override = {
                int(gid): np.array([0.0, 0.0, 0.0], dtype=float) for gid in original_list
            }
            orientation_mapping_pairs = [(g, g) for g in original_list]

        LOG.info(
            "[random-orientation] Mapped %d grain orientations (seed=%s)",
            len(grain_euler_override),
            seed,
        )

        if run_dir is not None and orientation_mapping_pairs:
            mapping_path = run_dir / "random_orientation_mapping.txt"
            with open(mapping_path, "w", encoding="utf-8") as f:
                f.write(
                    "# grain_id  orientation_from_grain_id  (grain_id uses Euler angles of orientation_from_grain_id)\n"
                )
                for gid, src_gid in orientation_mapping_pairs:
                    f.write(f"{gid}  {src_gid}\n")
            LOG.info("[random-orientation] Mapping saved to %s", mapping_path)

    tasks = [
        (
            int(g),
            str(dream3d),
            hx,
            hy,
            hz,
            lattice,
            ratio,
            extend,
            unit_extend_ratio,
            seed,
            voxel_csv_str,
            h5_grain_dset,
            h5_euler_dset,
            h5_numneighbors_dset,
            h5_neighborlist_dset,
            h5_dimensions_dset,
            grain_euler_override,
            h5_phases_dset,
            h5_phase_name_dset,
            effective_ratio,
        )
        for g in gids
    ]

    # Process grains with progress display
    import sys

    if workers <= 1:
        chunks: List[pd.DataFrame] = []
        LOG.info("[carve] Processing %d grains sequentially...", total_grains)
        sys.stdout.flush()  # Ensure output appears in SLURM logs immediately
        for i, t in enumerate(tasks, 1):
            grain_id = t[0]
            LOG.info("[carve] [%d/%d] Processing grain ID: %d", i, total_grains, grain_id)
            sys.stdout.flush()
            chunks.append(_carve_one(t))
            LOG.info("[carve] [%d/%d] ✓ Completed grain ID: %d", i, total_grains, grain_id)
            sys.stdout.flush()  # Flush after each grain for SLURM visibility
        df_all = pd.concat(chunks, ignore_index=True)
    else:
        LOG.info(
            "[carve] Processing %d grains with %d workers (mp_start=%s)...",
            total_grains,
            workers,
            mp_start,
        )
        if use_fork_preload:
            LOG.info(
                "[carve] Fork mode: using single Frame loaded in main (no per-worker HDF5 reload)."
            )
        else:
            LOG.info(
                "[carve] Spawn mode: each worker loads HDF5 once and reuses it for assigned grains."
            )
            LOG.info(
                "[carve] Note: Each worker holds a full copy of the HDF5 data. "
                "For very large files use --workers 1 or --mp-start fork on Linux."
            )
        sys.stdout.flush()
        if use_fork_preload:
            with ctx.Pool(processes=workers) as pool:
                chunks = []
                completed = 0
                for result in pool.imap(_carve_one, tasks):
                    completed += 1
                    grain_id = tasks[completed - 1][0]
                    percentage = int(100 * completed / total_grains)
                    LOG.info(
                        "[carve] [%d/%d] (%d%%) ✓ Completed grain ID: %d",
                        completed,
                        total_grains,
                        percentage,
                        grain_id,
                    )
                    sys.stdout.flush()
                    chunks.append(result)
                df_all = pd.concat(chunks, ignore_index=True)
        else:
            with ctx.Pool(
                processes=workers,
                initializer=_init_worker_frame,
                initargs=_init_args,
            ) as pool:
                chunks = []
                completed = 0
                for result in pool.imap(_carve_one, tasks):
                    completed += 1
                    grain_id = tasks[completed - 1][0]
                    percentage = int(100 * completed / total_grains)
                    LOG.info(
                        "[carve] [%d/%d] (%d%%) ✓ Completed grain ID: %d",
                        completed,
                        total_grains,
                        percentage,
                        grain_id,
                    )
                    sys.stdout.flush()
                    chunks.append(result)
                df_all = pd.concat(chunks, ignore_index=True)

    LOG.info("[done] merged %d grains → %d rows", len(gids), len(df_all))
    return df_all


# --------------------------- CLI wiring ---------------------------


def build_parser() -> argparse.ArgumentParser:
    def error_handler(message: str) -> None:
        print(f"\nError: {message}", file=sys.stderr)
        sys.exit(2)

    p = argparse.ArgumentParser(
        prog="expoly",
        description="EXPoly: Convert Dream3D voxel data to MD-ready atomistic structures (LAMMPS data files).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.error = error_handler
    sub = p.add_subparsers(dest="command", required=True, help="Available commands")

    # ==================== run command ====================
    r = sub.add_parser("run", help="Run carve + polish pipeline")

    # Input group
    input_group = r.add_argument_group("Input")
    input_group.add_argument(
        "--dream3d", type=Path, required=True, help="Path to Dream3D HDF5 file (.dream3d)"
    )
    input_group.add_argument(
        "--voxel-csv",
        type=Path,
        default=None,
        help="Optional voxel grid CSV (whitespace-separated). "
        "If provided, use VoxelCSVFrame (CSV grid + HDF5 orientations)",
    )
    input_group.add_argument(
        "--h5-grain-dset",
        type=str,
        default=None,
        help="Name of grain-ID dataset in HDF5 (default: FeatureIds). Example: GrainID",
    )
    input_group.add_argument(
        "--h5-euler-dset",
        type=str,
        default=None,
        help="Name of Euler angles dataset in HDF5 (default: EulerAngles)",
    )
    input_group.add_argument(
        "--h5-numneighbors-dset",
        type=str,
        default=None,
        help="Name of NumNeighbors dataset in HDF5 (default: NumNeighbors)",
    )
    input_group.add_argument(
        "--h5-neighborlist-dset",
        type=str,
        default=None,
        help="Name of NeighborList dataset in HDF5 (default: NeighborList). Example: NeighborList2",
    )
    input_group.add_argument(
        "--h5-dimensions-dset",
        type=str,
        default=None,
        help="Name of DIMENSIONS dataset in HDF5 (default: DIMENSIONS)",
    )
    input_group.add_argument(
        "--h5-phases-dset",
        type=str,
        default=None,
        help="Name of Phases dataset in HDF5 for multi-phase (default: Phases). "
        "If absent, phase-based lattice is disabled.",
    )
    input_group.add_argument(
        "--h5-phase-name-dset",
        type=str,
        default=None,
        help="Name of PhaseName dataset in HDF5 for multi-phase (default: PhaseName). "
        "If absent, phase-based lattice is disabled.",
    )

    # Region selection group
    region_group = r.add_argument_group("Region Selection")
    region_group.add_argument(
        "--hx",
        type=_parse_range,
        required=True,
        help="HX range in voxel space, e.g. 0:50 (inclusive)",
    )
    region_group.add_argument(
        "--hy",
        type=_parse_range,
        required=True,
        help="HY range in voxel space, e.g. 0:50 (inclusive)",
    )
    region_group.add_argument(
        "--hz",
        type=_parse_range,
        required=True,
        help="HZ range in voxel space, e.g. 0:50 (inclusive)",
    )

    # Carving group
    carve_group = r.add_argument_group("Carving")
    carve_group.add_argument(
        "--lattice",
        choices=["FCC", "BCC", "DIA"],
        default="FCC",
        help="Lattice type: FCC (Face-Centered cubic), BCC (Body-Centered cubic), DIA (diamond) (default: FCC)",
    )
    carve_group.add_argument(
        "--ratio",
        type=float,
        default=1.5,
        help="Lattice-to-voxel scale ratio (default: 1.5). Larger ratio → smaller grain size",
    )
    carve_group.add_argument(
        "--lattice-constant",
        type=_parse_lattice_constant,
        required=True,
        help="Physical lattice constant(s) in Å. Single phase: 3.524. Multi-phase: FCC:3.524,BCC:2.87",
    )
    carve_group.add_argument(
        "--extend", action="store_true", help="Use extended-neighborhood pipeline for carving"
    )
    carve_group.add_argument(
        "--unit-extend-ratio",
        type=int,
        default=3,
        help="Unit extend ratio for extended pipeline (default: 3, recommend odd numbers)",
    )
    carve_group.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Parallel workers for carving (default: 2)",
    )
    carve_group.add_argument(
        "--mp-start",
        type=str,
        default="auto",
        choices=("auto", "fork", "spawn"),
        help="Multiprocessing start method: auto (fork on Linux, spawn elsewhere), fork (share loaded Frame, no per-worker HDF5 reload), spawn (each worker loads HDF5; default on non-Linux)",
    )
    carve_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible carving (default: None)",
    )
    carve_group.add_argument(
        "--random-orientation",
        action="store_true",
        help="Randomize grain orientations only among grains in the selected H range (--hx/--hy/--hz). "
        "Each grain in that range gets a random orientation from another grain in the same range. "
        "Mapping is saved to random_orientation_mapping.txt in the run folder. Use --seed for reproducibility.",
    )

    # Polish group
    polish_group = r.add_argument_group("Polish")
    polish_group.add_argument(
        "--ovito-cutoff",
        type=float,
        default=1.6,
        help="OVITO overlap cutoff distance in Å (default: 1.6, safe for Ni FCC)",
    )
    polish_group.add_argument(
        "--atom-mass",
        type=float,
        default=58.6934,
        help="Atom mass for LAMMPS 'Masses' section (default: 58.6934, Ni)",
    )

    # Output group
    output_group = r.add_argument_group("Output")
    output_group.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Root output directory (default: runs/expoly-<timestamp>)",
    )
    output_group.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary files (tmp_polish.in.data, ovito_cleaned.data, overlap_mask.txt)",
    )
    output_group.add_argument(
        "--final-with-grain",
        action="store_true",
        help="Write additional final.dump with per-atom grain-ID",
    )
    output_group.add_argument(
        "--generate-voronoi",
        action="store_true",
        help="One-shot Voronoi refinement: run carve+polish → extract Voronoi CSV → run again with that CSV. "
        "Skips manual 'expoly voronoi' and second 'expoly run --voxel-csv'. Uses --voronoi-voxel-size for the middle step.",
    )
    output_group.add_argument(
        "--voronoi-voxel-size",
        type=float,
        default=2.0,
        help="Voxel size for Voronoi step when using --generate-voronoi (default: 2.0)",
    )

    # General options
    r.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    # ==================== voronoi command ====================
    v = sub.add_parser(
        "voronoi", help="Extract GB topology from LAMMPS dump and generate voxel_all.csv"
    )
    v.add_argument(
        "--dump", type=Path, required=True, help="Path to LAMMPS dump file (one timestep)"
    )
    v.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., check_large_voxel_from_mesh_what.csv)",
    )
    v.add_argument(
        "--cube-ratio",
        type=float,
        default=0.015,
        dest="crop_ratio",
        help="Crop ratio per side (default 0.015)",
    )
    v.add_argument("--k", type=int, default=25, help="k-NN for classification (default 25)")
    v.add_argument(
        "--min-other-atoms", type=int, default=4, help="Min other-grain neighbors (default 4)"
    )
    v.add_argument(
        "--voxel-size", type=float, default=2.0, dest="voxel_size", help="Voxel size (default 2.0)"
    )
    v.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # ==================== doctor command ====================
    d = sub.add_parser("doctor", help="Validate input files and configuration")
    d.add_argument(
        "--dream3d", type=Path, required=True, help="Path to Dream3D HDF5 file to validate"
    )
    d.add_argument(
        "--h5-grain-dset",
        type=str,
        default=None,
        help="Name of grain-ID dataset (default: FeatureIds)",
    )
    d.add_argument(
        "--h5-euler-dset",
        type=str,
        default=None,
        help="Name of Euler angles dataset (default: EulerAngles)",
    )
    d.add_argument(
        "--h5-numneighbors-dset",
        type=str,
        default=None,
        help="Name of NumNeighbors dataset (default: NumNeighbors)",
    )
    d.add_argument(
        "--h5-neighborlist-dset",
        type=str,
        default=None,
        help="Name of NeighborList dataset (default: NeighborList)",
    )
    d.add_argument(
        "--h5-dimensions-dset",
        type=str,
        default=None,
        help="Name of DIMENSIONS dataset (default: DIMENSIONS)",
    )
    d.add_argument(
        "--h5-phases-dset",
        type=str,
        default=None,
        help="Name of Phases dataset for multi-phase (default: Phases)",
    )
    d.add_argument(
        "--h5-phase-name-dset",
        type=str,
        default=None,
        help="Name of PhaseName dataset for multi-phase (default: PhaseName)",
    )
    d.add_argument(
        "--hx", type=_parse_range, default=None, help="HX range to validate (optional, e.g. 0:50)"
    )
    d.add_argument(
        "--hy", type=_parse_range, default=None, help="HY range to validate (optional, e.g. 0:50)"
    )
    d.add_argument(
        "--hz", type=_parse_range, default=None, help="HZ range to validate (optional, e.g. 0:50)"
    )
    d.add_argument(
        "--check-ovito", action="store_true", help="Check if OVITO is installed and importable"
    )
    d.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    return p


def run_noninteractive(ns: argparse.Namespace, run_dir: Path | None = None) -> int:
    _init_logging(ns.verbose)

    if run_dir is None:
        run_dir = _mk_run_dir(ns.outdir)
    (hx0, hx1), (hy0, hy1), (hz0, hz1) = ns.hx, ns.hy, ns.hz

    # Normalize lattice_constant: CLI gives dict; pipeline API may give float
    lc_raw = ns.lattice_constant
    if isinstance(lc_raw, (int, float)):
        lattice_constants = {"FCC": float(lc_raw)}
    else:
        lattice_constants = lc_raw

    # 1) carve (extendable)
    df_all = _carve_all(
        dream3d=ns.dream3d,
        hx=(hx0, hx1),
        hy=(hy0, hy1),
        hz=(hz0, hz1),
        lattice=ns.lattice,
        ratio=ns.ratio,
        extend=ns.extend,
        unit_extend_ratio=ns.unit_extend_ratio,
        workers=int(ns.workers),
        seed=ns.seed,
        voxel_csv=ns.voxel_csv,
        h5_grain_dset=ns.h5_grain_dset,
        h5_euler_dset=getattr(ns, "h5_euler_dset", None),
        h5_numneighbors_dset=getattr(ns, "h5_numneighbors_dset", None),
        h5_neighborlist_dset=getattr(ns, "h5_neighborlist_dset", None),
        h5_dimensions_dset=getattr(ns, "h5_dimensions_dset", None),
        h5_phases_dset=getattr(ns, "h5_phases_dset", None),
        h5_phase_name_dset=getattr(ns, "h5_phase_name_dset", None),
        lattice_constants=lattice_constants,
        random_orientation=getattr(ns, "random_orientation", False),
        run_dir=run_dir,
        mp_start=getattr(ns, "mp_start", "auto"),
    )

    raw_points_path = run_dir / "raw_points.parquet"
    df_all.to_parquet(raw_points_path, index=False)
    LOG.info("[done] raw points → %s (rows=%d)", raw_points_path, len(df_all))

    # 2) polish (OVITO required)
    #    With effective_ratio (multi-phase): unified scan_ratio = ref_lat/ratio for all phases
    #    Single-phase: scan_ratio = lattice_constant / ratio
    lc = lattice_constants
    ratio = float(ns.ratio)
    if len(lc) == 1:
        ((_, val),) = lc.items()
        scan_ratio = val / ratio
        lattice_scan_ratios = None
        LOG.info(
            "[polish] lattice_constant=%.6g, cube_ratio=%.6g → scan_ratio=%.6g",
            val,
            ratio,
            scan_ratio,
        )
    else:
        ref_lat = min(lc.values())
        scan_ratio = ref_lat / ratio
        lattice_scan_ratios = None
        LOG.info(
            "[polish] multi-phase (unified physical scale): ref_lat=%.6g, cube_ratio=%.6g → scan_ratio=%.6g",
            ref_lat,
            ratio,
            scan_ratio,
        )

    pcfg = PolishConfig(
        scan_ratio=scan_ratio,
        lattice_scan_ratios=lattice_scan_ratios,
        cube_ratio=float(ns.ratio),
        hx_range=(hx0, hx1),
        hy_range=(hy0, hy1),
        hz_range=(hz0, hz1),
        real_extent=bool(ns.extend),  # Auto-enable real_extent when extend is used
        unit_extend_ratio=int(ns.unit_extend_ratio),
        ovito_cutoff=float(ns.ovito_cutoff),
        atom_mass=float(ns.atom_mass),
        keep_tmp=bool(ns.keep_tmp),
        overwrite=True,
    )

    paths = {
        "tmp_in": run_dir / "tmp_polish.in.data",
        "ovito_mask": run_dir / "overlap_mask.npy",
        "ovito_psc": run_dir / "ovito_cleaned.data",
        "final_lmp": run_dir / "final.data",
    }

    final_path = polish_pipeline(
        raw_points_path,
        pcfg,
        paths,
        final_with_grain=bool(ns.final_with_grain),
    )

    LOG.info("All done. final → %s", final_path)
    return 0


def voronoi_command(ns: argparse.Namespace) -> int:
    """Run Voronoi topology extraction from LAMMPS dump and generate voxel_all.csv."""
    _init_logging(ns.verbose)
    out = voronoi_run(
        dump_path=ns.dump,
        output_path=ns.output,
        crop_ratio=getattr(ns, "crop_ratio", 0.015),
        k=getattr(ns, "k", 25),
        min_other_atoms=getattr(ns, "min_other_atoms", 4),
        voxel_size=getattr(ns, "voxel_size", 2.0),
    )
    LOG.info("Voronoi output → %s", out)
    return 0


def doctor_command(ns: argparse.Namespace) -> int:
    """Run doctor command to validate inputs."""
    _init_logging(ns.verbose)

    issues = []
    warnings = []
    info = []

    # Check file existence
    dream3d_path = Path(ns.dream3d)
    if not dream3d_path.exists():
        issues.append(f"✗ Dream3D file not found: {dream3d_path}")
        print("\n".join(issues))
        return 1
    else:
        info.append(f"✓ Dream3D file exists: {dream3d_path}")

    # Try to load Frame
    frame = None
    # Build mapping with custom dataset names or defaults
    grain_dset = ns.h5_grain_dset or "FeatureIds"
    euler_dset = getattr(ns, "h5_euler_dset", None) or "EulerAngles"
    numneighbors_dset = getattr(ns, "h5_numneighbors_dset", None) or "NumNeighbors"
    neighborlist_dset = getattr(ns, "h5_neighborlist_dset", None) or "NeighborList"
    dimensions_dset = getattr(ns, "h5_dimensions_dset", None) or "DIMENSIONS"
    mapping = {
        "GrainId": grain_dset,
        "Euler": euler_dset,
        "Num_NN": numneighbors_dset,
        "Num_list": neighborlist_dset,
        "Dimension": dimensions_dset,
    }
    # Map attribute names to CLI argument names for error messages
    attr_to_arg = {
        "GrainId": "--h5-grain-dset",
        "Euler": "--h5-euler-dset",
        "Num_NN": "--h5-numneighbors-dset",
        "Num_list": "--h5-neighborlist-dset",
        "Dimension": "--h5-dimensions-dset",
    }

    # Build mapping with custom dataset names or defaults
    h5_phases_dset = getattr(ns, "h5_phases_dset", None)
    h5_phase_name_dset = getattr(ns, "h5_phase_name_dset", None)
    try:
        frame = Frame(
            str(dream3d_path),
            mapping=mapping,
            h5_phases_dset=h5_phases_dset,
            h5_phase_name_dset=h5_phase_name_dset,
        )
        info.append("✓ Successfully loaded HDF5 file")
        info.append(
            f"  Volume dimensions: HX=[0,{frame.HX_lim}), HY=[0,{frame.HY_lim}), HZ=[0,{frame.HZ_lim})"
        )

        # Check grain IDs
        unique_gids = np.unique(frame.fid)
        positive_gids = unique_gids[unique_gids > 0]
        info.append(f"  Found {len(positive_gids)} positive grain IDs (excluding void=0)")
        if len(positive_gids) == 0:
            warnings.append("⚠ No positive grain IDs found in the volume")

    except KeyError as e:
        dataset_name = str(e).strip("'\"")
        # Find which attribute failed
        failed_attr = None
        for attr, ds_name in mapping.items():
            if ds_name == dataset_name:
                failed_attr = attr
                break
        arg_name = attr_to_arg.get(failed_attr, "--h5-*-dset") if failed_attr else "--h5-*-dset"
        issues.append(
            f"✗ Missing dataset '{dataset_name}' in HDF5 file.\n"
            f"  Solution: Use {arg_name} to specify a different dataset name, "
            f"or check the HDF5 structure with 'h5dump -n {dream3d_path}'"
        )
    except Exception as e:
        issues.append(f"✗ Failed to load HDF5 file: {e}")

    # Validate H ranges if provided
    if ns.hx and ns.hy and ns.hz and frame is not None:
        try:
            hx0, hx1 = ns.hx
            hy0, hy1 = ns.hy
            hz0, hz1 = ns.hz

            if hx0 < 0 or hx1 > frame.HX_lim:
                issues.append(
                    f"✗ HX range [{hx0}, {hx1}] is outside volume bounds [0, {frame.HX_lim})"
                )
            else:
                info.append(f"✓ HX range [{hx0}, {hx1}] is within bounds [0, {frame.HX_lim})")

            if hy0 < 0 or hy1 > frame.HY_lim:
                issues.append(
                    f"✗ HY range [{hy0}, {hy1}] is outside volume bounds [0, {frame.HY_lim})"
                )
            else:
                info.append(f"✓ HY range [{hy0}, {hy1}] is within bounds [0, {frame.HY_lim})")

            if hz0 < 0 or hz1 > frame.HZ_lim:
                issues.append(
                    f"✗ HZ range [{hz0}, {hz1}] is outside volume bounds [0, {frame.HZ_lim})"
                )
            else:
                info.append(f"✓ HZ range [{hz0}, {hz1}] is within bounds [0, {frame.HZ_lim})")

            # Check if grains exist in range
            if not issues:  # Only if ranges are valid
                try:
                    gids = frame.find_volume_grain_ID(
                        (hx0, hx1), (hy0, hy1), (hz0, hz1), return_count=False
                    )
                    positive_gids = gids[gids > 0]
                    if len(positive_gids) > 0:
                        info.append(
                            f"✓ Found {len(positive_gids)} positive grain IDs in specified H ranges"
                        )
                    else:
                        warnings.append(
                            f"⚠ No positive grain IDs found in H ranges [{hx0}:{hx1}, {hy0}:{hy1}, {hz0}:{hz1}]"
                        )
                except Exception as e:
                    warnings.append(f"⚠ Could not check grains in H ranges: {e}")
        except NameError:
            pass  # Frame not loaded, skip range validation

    # Check OVITO if requested
    if ns.check_ovito:
        try:
            from ovito.io import import_file  # noqa: F401

            info.append("✓ OVITO is installed and importable")
        except ImportError:
            issues.append("✗ OVITO is not installed. Install with: pip install ovito")

    # Print results
    print("EXPoly Doctor - Input Validation\n" + "=" * 60)

    if info:
        print("\n[INFO]")
        for msg in info:
            print(f"  {msg}")

    if warnings:
        print("\n[WARNINGS]")
        for msg in warnings:
            print(f"  {msg}")

    if issues:
        print("\n[ISSUES]")
        for msg in issues:
            print(f"  {msg}")
        print("\n" + "=" * 60)
        print("✗ Validation failed. Please fix the issues above.")
        return 1
    else:
        print("\n" + "=" * 60)
        print("✓ All checks passed!")
        return 0


def main(argv: Iterable[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        ns, unknown = parser.parse_known_args(argv)
        if unknown:
            print(f"\nError: unrecognized arguments: {' '.join(unknown)}", file=sys.stderr)
            print(f"Full command line: {' '.join(sys.argv)}", file=sys.stderr)
            parser.print_help()
            return 2
    except SystemExit as e:
        return e.code if e.code is not None else 2
    except Exception as e:
        print(f"\nUnexpected error parsing arguments: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"Command line args: {sys.argv}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        parser.print_help()
        return 2

    if ns.command == "run":
        if getattr(ns, "generate_voronoi", False):
            run_dir = _mk_run_dir(ns.outdir)
            old_final_grain = ns.final_with_grain
            ns.voxel_csv = None
            ns.final_with_grain = True
            run_noninteractive(ns, run_dir=run_dir)
            voronoi_csv = run_dir / "voronoi.csv"
            voronoi_run(
                dump_path=run_dir / "final.dump",
                output_path=voronoi_csv,
                crop_ratio=0.015,
                k=25,
                min_other_atoms=4,
                voxel_size=float(getattr(ns, "voronoi_voxel_size", 2.0)),
            )
            LOG.info("Voronoi CSV → %s; running second pass with --voxel-csv", voronoi_csv)
            ns.voxel_csv = voronoi_csv
            ns.final_with_grain = old_final_grain
            # Second pass uses voxel grid from CSV: set hx/hy/hz to CSV extent to avoid IndexError
            (ns.hx, ns.hy, ns.hz) = _voxel_csv_h_ranges(voronoi_csv)
            LOG.info(
                "Second pass H ranges from voronoi grid: hx=%s hy=%s hz=%s",
                ns.hx,
                ns.hy,
                ns.hz,
            )
            return run_noninteractive(ns, run_dir=run_dir)
        return run_noninteractive(ns)
    elif ns.command == "voronoi":
        return voronoi_command(ns)
    elif ns.command == "doctor":
        return doctor_command(ns)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
