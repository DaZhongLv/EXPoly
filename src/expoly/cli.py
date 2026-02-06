# src/expoly/cli.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from expoly.carve import CarveConfig, process, process_extend
from expoly.frames import Frame, VoxelCSVFrame
from expoly.polish import PolishConfig, polish_pipeline

LOG = logging.getLogger("expoly.cli")


# --------------------------- small helpers ---------------------------


def _parse_range(s: str) -> Tuple[int, int]:
    """
    Parse 'a:b' (also tolerates '[a:b]', spaces, etc.) → (a, b), both int.
    """
    s = s.strip().lstrip("[").rstrip("]").strip()
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"Range must be like '0:50', got {s!r}")
    a, b = s.split(":", 1)
    return (int(a.strip()), int(b.strip()))


def _mk_run_dir(root: Path | None = None) -> Path:
    ts = int(time.time())
    base = Path("runs") if root is None else Path(root)
    path = base / f"expoly-{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _init_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s: %(message)s",
        force=True,  # Override any existing configuration
    )
    # Ensure output is flushed immediately (important for SLURM/supercomputers)
    # This ensures progress messages appear in SLURM output files in real-time
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Fallback if reconfigure not available
    # Ensure output is flushed immediately (important for SLURM/supercomputers)
    import sys

    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


# --------------------------- grain selection ---------------------------


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
        if voxel_csv is None:
            # Pure Dream3D path with customizable dataset names
            return Frame(str(dream3d_path), mapping=mapping)
        else:
            if not Path(voxel_csv).exists():
                raise FileNotFoundError(
                    f"Voxel CSV file not found: {voxel_csv}. " f"Please check the file path."
                )
            # Voxel-CSV + HDF5 combination
            return VoxelCSVFrame(
                path=str(dream3d_path),
                voxel_csv=str(voxel_csv),
                h5_grain_dset=mapping["GrainId"],
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
        )
    frame = _worker_frame_cache

    ccfg = CarveConfig(
        lattice=lattice,
        ratio=float(ratio),
        unit_extend_ratio=int(unit_extend_ratio),
        rng_seed=None if seed is None else int(seed),
    )

    if extend:
        df = process_extend(grain_id, frame, ccfg, grain_euler_override=grain_euler_override)
    else:
        df = process(grain_id, frame, ccfg, grain_euler_override=grain_euler_override)

    # Required column order: X,Y,Z,HX,HY,HZ,margin-ID,grain-ID
    cols = ["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID", "grain-ID"]
    df = df[cols].copy()
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
    random_orientation: bool = False,
) -> pd.DataFrame:
    frame = _build_frame_for_carve(
        dream3d,
        voxel_csv=voxel_csv,
        h5_grain_dset=h5_grain_dset,
        h5_euler_dset=h5_euler_dset,
        h5_numneighbors_dset=h5_numneighbors_dset,
        h5_neighborlist_dset=h5_neighborlist_dset,
        h5_dimensions_dset=h5_dimensions_dset,
    )
    gids = _pick_grain_ids(frame, hx, hy, hz)
    total_grains = len(gids)
    LOG.info("carve: %d grains selected in H ranges HX=%s HY=%s HZ=%s", total_grains, hx, hy, hz)

    # Build grain→Euler override map for random-orientation mode
    grain_euler_override: Dict[int, np.ndarray] | None = None
    if random_orientation:
        LOG.info("[random-orientation] Building shuffled orientation mapping...")
        # Original list: all selected grain IDs in order
        original_list = [int(g) for g in gids]
        # Shuffled list: same IDs but shuffled
        shuffled_list = original_list.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(shuffled_list)

        # Optimized: batch compute all grain orientations at once using pandas groupby
        # Instead of calling search_avg_Euler 437 times (each scans entire fid array),
        # compute all averages in one pass
        import pandas as pd

        gids_set = set(original_list)
        mask_selected = np.isin(frame.fid, original_list)
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

            # Build mapping: for each grain_id, find its position in shuffled_list,
            # then use that position to get the grain_id from original_list,
            # and get Euler angle from precomputed grain_means
            grain_euler_override = {}
            for grain_id in original_list:
                # Find grain_id's position in shuffled_list
                shuffled_index = shuffled_list.index(grain_id)
                # Get the grain_id at that position in original_list
                euler_source_grain_id = original_list[shuffled_index]
                # Get Euler angle from precomputed means
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
            # Fallback: no selected grains found
            LOG.warning("[random-orientation] No selected grains found in data")
            grain_euler_override = {
                int(gid): np.array([0.0, 0.0, 0.0], dtype=float) for gid in original_list
            }

        LOG.info(
            "[random-orientation] Mapped %d grain orientations (seed=%s)",
            len(grain_euler_override),
            seed,
        )

    voxel_csv_str = str(voxel_csv) if voxel_csv is not None else None
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
        import multiprocessing as mp

        LOG.info("[carve] Processing %d grains with %d workers...", total_grains, workers)
        LOG.info(
            "[carve] Each worker will load HDF5 file once and reuse it for all assigned grains"
        )
        sys.stdout.flush()
        # Initialize each worker process with Frame loaded once (shared across tasks in that worker)
        init_args = (
            str(dream3d),
            voxel_csv_str,
            h5_grain_dset,
            h5_euler_dset,
            h5_numneighbors_dset,
            h5_neighborlist_dset,
            h5_dimensions_dset,
        )
        with mp.get_context("spawn").Pool(
            processes=workers, initializer=_init_worker_frame, initargs=init_args
        ) as pool:
            # Use imap for progress tracking
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
                sys.stdout.flush()  # Ensure progress appears in SLURM output files in real-time
                chunks.append(result)
        df_all = pd.concat(chunks, ignore_index=True)

    LOG.info("[done] merged %d grains → %d rows", len(gids), len(df_all))
    return df_all


# --------------------------- CLI wiring ---------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="expoly",
        description="EXPoly: Convert Dream3D voxel data to MD-ready atomistic structures (LAMMPS data files).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        help="Name of grain-ID dataset in HDF5 (default: FeatureIds). " "Example: GrainID",
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
        help="Name of NeighborList dataset in HDF5 (default: NeighborList). "
        "Example: NeighborList2",
    )
    input_group.add_argument(
        "--h5-dimensions-dset",
        type=str,
        default=None,
        help="Name of DIMENSIONS dataset in HDF5 (default: DIMENSIONS)",
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
        help="Lattice-to-voxel scale ratio (default: 1.5). " "Larger ratio → smaller grain size",
    )
    carve_group.add_argument(
        "--lattice-constant",
        type=float,
        required=True,
        help="Physical lattice constant in Å (e.g., 3.524 for Ni)",
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
        default=os.cpu_count() or 1,
        help="Parallel workers for carving (default: CPU count)",
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
        help="Randomize grain orientations: shuffle grain IDs and reassign orientations. "
        "Each grain ID gets a random orientation from the shuffled list. Use --seed for reproducibility.",
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

    # General options
    r.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

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


def run_noninteractive(ns: argparse.Namespace) -> int:
    _init_logging(ns.verbose)

    run_dir = _mk_run_dir(ns.outdir)
    (hx0, hx1), (hy0, hy1), (hz0, hz1) = ns.hx, ns.hy, ns.hz

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
        random_orientation=getattr(ns, "random_orientation", False),
    )

    raw_csv = run_dir / "raw_points.csv"
    df_all.to_csv(raw_csv, header=False, index=False, sep=" ")
    LOG.info("[done] raw points → %s (rows=%d)", raw_csv, len(df_all))

    # 2) polish (OVITO required)
    #    scan_ratio = lattice_constant / cube_ratio; here cube_ratio is --ratio
    scan_ratio = float(ns.lattice_constant) / float(ns.ratio)
    LOG.info(
        "[polish] lattice_constant=%.6g, cube_ratio=%.6g → scan_ratio=%.6g",
        ns.lattice_constant,
        ns.ratio,
        scan_ratio,
    )

    pcfg = PolishConfig(
        scan_ratio=scan_ratio,
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
        "ovito_mask": run_dir / "overlap_mask.txt",
        "ovito_psc": run_dir / "ovito_cleaned.data",
        "final_lmp": run_dir / "final.data",
    }

    final_path = polish_pipeline(
        raw_csv,
        pcfg,
        paths,
        final_with_grain=bool(ns.final_with_grain),
    )

    LOG.info("All done. final → %s", final_path)
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
    try:
        frame = Frame(str(dream3d_path), mapping=mapping)
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
    ns = parser.parse_args(argv)

    if ns.command == "run":
        return run_noninteractive(ns)
    elif ns.command == "doctor":
        return doctor_command(ns)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
