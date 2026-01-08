# src/expoly/cli.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from expoly.frames import Frame, VoxelCSVFrame
from expoly.carve import CarveConfig, process, process_extend
from expoly.polish import PolishConfig, polish_pipeline

LOG = logging.getLogger("expoly.cli")


# --------------------------- small helpers ---------------------------

def _parse_range(s: str) -> Tuple[int, int]:
    """
    Parse 'a:b' (也容忍 '[a:b]'、空格等) → (a, b)，均为 int。
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
        format="%(levelname)s | %(name)s: %(message)s"
    )

# --------------------------- grain selection ---------------------------

def _pick_grain_ids(f: Frame, hx: Tuple[int, int], hy: Tuple[int, int], hz: Tuple[int, int]) -> np.ndarray:
    """
    用 Dream3D 的体素范围挑 grain（>0）。
    """
    gids = f.find_volume_grain_ID(hx, hy, hz, return_count=False)
    gids = np.asarray(gids, dtype=int)
    gids = gids[gids > 0]
    if gids.size == 0:
        raise RuntimeError("No positive grain id found within the provided H ranges.")
    return np.unique(gids)


# --------------------------- frame builder ---------------------------

def _build_frame_for_carve(
    dream3d_path: Path | str,
    voxel_csv: Path | None,
    h5_grain_dset: str | None,
) -> Frame:
    """
    根据是否提供 voxel_csv 来返回 Frame 或 VoxelCSVFrame。
    h5_grain_dset 用来覆盖 HDF5 中 grain-ID 数据集名（默认 FeatureIds）。
    """
    dream3d_path = Path(dream3d_path)

    grain_dset = h5_grain_dset or "FeatureIds"

    if voxel_csv is None:
        # 纯 Dream3D 路径（旧逻辑），但允许自定义 grain dset 名
        mapping = {
            "GrainId": grain_dset,
            "Euler": "EulerAngles",
            "Num_NN": "NumNeighbors",
            "Num_list": "NeighborList",
            "Dimension": "DIMENSIONS",
        }
        return Frame(str(dream3d_path), mapping=mapping)
    else:
        # 新逻辑：voxel-CSV + h5 组合
        return VoxelCSVFrame(
            path=str(dream3d_path),
            voxel_csv=str(voxel_csv),
            h5_grain_dset=grain_dset,
        )



# --------------------------- carve runner ---------------------------

def _carve_one(args) -> pd.DataFrame:
    """
    子进程调用器：根据 extend 选择流程；失败则抛出异常（主进程会记录）。
    """
    (grain_id, dream3d_path, hx, hy, hz, lattice, ratio, extend, unit_extend_ratio, seed, voxel_csv,
     h5_grain_dset) = args

    # 每个子进程自己打开 Frame（避免跨进程句柄问题）
    frame = _build_frame_for_carve(
        dream3d_path,
        voxel_csv=Path(voxel_csv) if voxel_csv is not None else None,
        h5_grain_dset=h5_grain_dset,
    )

    ccfg = CarveConfig(
        lattice=lattice,
        ratio=float(ratio),
        unit_extend_ratio=int(unit_extend_ratio),
        rng_seed=None if seed is None else int(seed),
    )

    if extend:
        df = process_extend(grain_id, frame, ccfg)
    else:
        df = process(grain_id, frame, ccfg)

    # 要求列顺序：X,Y,Z,HX,HY,HZ,margin-ID,grain-ID
    cols = ['X','Y','Z','HX','HY','HZ','margin-ID','grain-ID']
    df = df[cols].copy()
    return df

def _carve_all(
    dream3d: Path,
    hx: Tuple[int, int], hy: Tuple[int, int], hz: Tuple[int, int],
    lattice: str, ratio: float,
    extend: bool, unit_extend_ratio: int,
    workers: int, seed: int | None,
    voxel_csv: Path | None,
    h5_grain_dset: str | None,
) -> pd.DataFrame:
    frame = _build_frame_for_carve(
        dream3d,
        voxel_csv=voxel_csv,
        h5_grain_dset=h5_grain_dset,
    )
    gids = _pick_grain_ids(frame, hx, hy, hz)
    LOG.info("carve: %d grains selected in H ranges HX=%s HY=%s HZ=%s", len(gids), hx, hy, hz)

    voxel_csv_str = str(voxel_csv) if voxel_csv is not None else None
    tasks = [
        (
            int(g),
            str(dream3d),
            hx, hy, hz,
            lattice, ratio,
            extend, unit_extend_ratio,
            seed,
            voxel_csv_str,
            h5_grain_dset,
        )
        for g in gids
    ]


    if workers <= 1:
        chunks: List[pd.DataFrame] = []
        for t in tasks:
            chunks.append(_carve_one(t))
        df_all = pd.concat(chunks, ignore_index=True)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            chunks = pool.map(_carve_one, tasks)
        df_all = pd.concat(chunks, ignore_index=True)

    LOG.info("[done] merged %d grains → %d rows", len(gids), len(df_all))
    return df_all

# --------------------------- CLI wiring ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="expoly", description="EXPoly runner")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run carve + polish")
    r.add_argument("--dream3d", type=Path, required=True, help="Path to Dream3D file")
    r.add_argument("--voxel-csv",type=Path,default=None,help="Optional voxel grid CSV (whitespace-separated). "
             "If provided, use VoxelCSVFrame (CSV grid + HDF5 orientations).",)
    r.add_argument("--h5-grain-dset",type=str,default=None,help="Name of grain-ID dataset in HDF5 (default: FeatureIds). "
             "Example: GrainID.",
    )
    r.add_argument("--hx", type=_parse_range, required=True, help="HX range, e.g. 0:50")
    r.add_argument("--hy", type=_parse_range, required=True, help="HY range, e.g. 0:50")
    r.add_argument("--hz", type=_parse_range, required=True, help="HZ range, e.g. 0:50")

    r.add_argument("--lattice", choices=["FCC","BCC","DIA"], default="FCC", help="Lattice type")
    r.add_argument("--ratio", type=float, default=1.5, help="Cube ratio used during carving (a in H units)")
    r.add_argument("--lattice-constant", type=float, required=True, help="Physical lattice constant (Å)")

    r.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers for carving")
    r.add_argument("--seed", type=int, default=None, help="Random seed for carving order / ball center")

    # <<< 新增：extend 相关开关 >>>
    r.add_argument("--extend", action="store_true", help="Use extended-neighborhood pipeline for carving")
    r.add_argument("--unit-extend-ratio", type=int, default=3, help="Unit extend ratio (recommend odd numbers: 3,5,7...)")

    # <<< 新增：polish 端范围同步放大 >>>
    r.add_argument("--real-extent", action="store_true", help="Multiply HX/HY/HZ ranges by unit-extend-ratio in polish")

    # OVITO / 输出
    r.add_argument("--ovito-cutoff", type=float, default=1.6, help="Overlap cutoff distance in OVITO")
    r.add_argument("--atom-mass", type=float, default=58.6934, help="Mass for 'Masses' section in final.data")
    r.add_argument("--keep-tmp", action="store_true", help="Keep temporary files (tmp_in / ovito_psc / mask)")
    r.add_argument("--outdir", type=Path, default=None, help="Root output dir (default: runs/expoly-<ts>)")

    r.add_argument("--final-with-grain", action="store_true",
        help="Append per-atom grain-ID as an extra column in final.data (Atoms lines). "
         "Note: may not be strictly compatible with LAMMPS atom_style atomic."
    )

    r.add_argument("-v", "--verbose", action="store_true", help="Verbose logs")
    return p

def run_noninteractive(ns: argparse.Namespace) -> int:
    _init_logging(ns.verbose)

    run_dir = _mk_run_dir(ns.outdir)
    (hx0, hx1), (hy0, hy1), (hz0, hz1) = ns.hx, ns.hy, ns.hz

    # 1) carve（可扩展）
    df_all = _carve_all(
        dream3d=ns.dream3d,
        hx=(hx0, hx1), hy=(hy0, hy1), hz=(hz0, hz1),
        lattice=ns.lattice, ratio=ns.ratio,
        extend=ns.extend, unit_extend_ratio=ns.unit_extend_ratio,
        workers=int(ns.workers), seed=ns.seed,
        voxel_csv=ns.voxel_csv,
        h5_grain_dset=ns.h5_grain_dset,
    )

    raw_csv = run_dir / "raw_points.csv"
    df_all.to_csv(raw_csv, header=False, index=False)
    LOG.info("[done] raw points → %s (rows=%d)", raw_csv, len(df_all))

    # 2) polish（强制 OVITO）
    #    scan_ratio = lattice_constant / cube_ratio；此处 cube_ratio 就是 --ratio
    scan_ratio = float(ns.lattice_constant) / float(ns.ratio)
    LOG.info("[polish] lattice_constant=%.6g, cube_ratio=%.6g → scan_ratio=%.6g",
             ns.lattice_constant, ns.ratio, scan_ratio)

    pcfg = PolishConfig(
        scan_ratio=scan_ratio,
        cube_ratio=float(ns.ratio),
        hx_range=(hx0, hx1), hy_range=(hy0, hy1), hz_range=(hz0, hz1),
        real_extent=bool(ns.real_extent),
        unit_extend_ratio=int(ns.unit_extend_ratio),
        ovito_cutoff=float(ns.ovito_cutoff),
        atom_mass=float(ns.atom_mass),
        keep_tmp=bool(ns.keep_tmp),
        overwrite=True,
    )

    paths = {
        "tmp_in":     run_dir / "tmp_polish.in.data",
        "ovito_mask": run_dir / "overlap_mask.txt",
        "ovito_psc":  run_dir / "ovito_cleaned.data",
        "final_lmp":  run_dir / "final.data",
    }

    final_path = polish_pipeline(
        raw_csv,
        pcfg,
        paths,
        final_with_grain=bool(ns.final_with_grain),
    )

    LOG.info("All done. final → %s", final_path)
    return 0

def main(argv: Iterable[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "run":
        return run_noninteractive(ns)
    parser.print_help()
    return 2

if __name__ == "__main__":
    sys.exit(main())


