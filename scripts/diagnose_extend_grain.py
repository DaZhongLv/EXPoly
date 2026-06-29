#!/usr/bin/env python3
"""
Diagnose extend pipeline for a single grain ID.

Reports row counts at each step (same as process_extend) so you can see where
the table becomes empty before carve_points crashes.

Usage (on supercomputer, from repo root):

  python scripts/diagnose_extend_grain.py \\
    --dream3d /path/to/file.dream3d \\
    --voxel-csv /path/to/voxel.csv \\
    --grain-id 1234 \\
    --unit-extend-ratio 3

With explicit H ranges instead of auto from CSV:

  python scripts/diagnose_extend_grain.py \\
    --dream3d /path/to/file.dream3d \\
    --hx 0:170 --hy 0:140 --hz 0:610 \\
    --grain-id 1234

Optional: try carve_points (will show the same IndexError if extend is empty):

  python scripts/diagnose_extend_grain.py ... --try-carve
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running without pip install -e .
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from expoly.carve import CarveConfig, carve_points  # noqa: E402
from expoly.cli import (  # noqa: E402
    _build_frame_for_carve,
    _parse_range,
    _pick_grain_ids,
    _resolve_run_h_ranges,
)


def _summary(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n=== {name} ===")
    print(f"  rows: {n}")
    if n == 0:
        print("  (empty)")
        return
    print(f"  columns: {list(df.columns)}")
    if "margin-ID" in df.columns:
        vc = df["margin-ID"].value_counts().sort_index()
        print(f"  margin-ID counts: {vc.to_dict()}")
    if "grain-ID" in df.columns:
        ug = df["grain-ID"].nunique()
        print(f"  unique grain-ID: {ug} (sample: {sorted(df['grain-ID'].unique())[:8]})")
    for col in ("HX", "HY", "HZ"):
        if col in df.columns:
            print(f"  {col} range: [{df[col].min()}, {df[col].max()}]")


def diagnose(
    dream3d: Path,
    grain_id: int,
    hx: tuple[int, int] | None,
    hy: tuple[int, int] | None,
    hz: tuple[int, int] | None,
    voxel_csv: Path | None,
    unit_extend_ratio: int,
    h5_neighborlist_dset: str | None,
    h5_grain_dset: str | None,
    try_carve: bool,
) -> int:
    hx_r, hy_r, hz_r = _resolve_run_h_ranges(voxel_csv, hx, hy, hz)
    print(f"dream3d: {dream3d}")
    if voxel_csv:
        print(f"voxel-csv: {voxel_csv}")
    print(f"H ranges: HX={hx_r} HY={hy_r} HZ={hz_r}")
    print(f"grain-id: {grain_id}")
    print(f"unit-extend-ratio: {unit_extend_ratio}")

    frame = _build_frame_for_carve(
        dream3d,
        voxel_csv,
        h5_grain_dset=h5_grain_dset,
        h5_neighborlist_dset=h5_neighborlist_dset,
    )
    print(f"volume limits: HX=[0,{frame.HX_lim}) HY=[0,{frame.HY_lim}) HZ=[0,{frame.HZ_lim})")

    gids = _pick_grain_ids(frame, hx_r, hy_r, hz_r)
    print(f"grains in H range: {len(gids)}")
    if grain_id not in gids:
        print(f"\nWARNING: grain {grain_id} is NOT in the selected H-range grain list.")
        print("  Extend may still run (uses full volume), but this ID was not found in the crop.")
    else:
        idx = int(list(gids).index(grain_id)) + 1
        print(f"  grain {grain_id} is #{idx} in sorted unique gids (of {len(gids)})")

    # Step 0: raw grain voxels
    out = frame.from_ID_to_D(grain_id)
    _summary("0) from_ID_to_D (grain voxels)", out)

    # Step 1: margin field
    out_margin = frame.find_grain_NN_with_out(grain_id)
    _summary("1) find_grain_NN_with_out", out_margin)

    if len(out_margin) == 0:
        print("\n>>> STOP: out_margin is empty — extend will fail here.")
        return 1

    # Step 2: extend (before renew_outer_margin)
    extend_raw = frame.get_extend_Out_(out_margin, unit_extend_ratio)
    _summary("2) get_extend_Out_", extend_raw)

    # Step 3: renew outer margin
    extend = frame.renew_outer_margin(extend_raw)
    _summary("3) renew_outer_margin", extend)

    extend_xyz = extend.rename(columns={"grain-ID": "ID"}).copy()
    _summary("4) extend_xyz (input to carve_points)", extend_xyz)

    if len(extend_xyz) == 0:
        print("\n>>> FAIL: extend_xyz is EMPTY — expoly run --extend will crash with:")
        print("    IndexError: single positional indexer is out-of-bounds")
        print("    (prepare_carve_meta: out_df['ID'].iloc[0])")
        return 1

    print("\n>>> OK: extend_xyz has rows — this grain should pass the iloc[0] check.")

    if try_carve:
        print("\n--- trying carve_points (may take a while) ---")
        cfg = CarveConfig(unit_extend_ratio=unit_extend_ratio, ratio=1.5, lattice="FCC")
        try:
            pts = carve_points(extend_xyz, frame, cfg)
            print(f"carve_points returned {len(pts)} lattice points")
        except Exception as e:
            print(f"carve_points FAILED: {type(e).__name__}: {e}")
            return 1

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose extend pipeline for one grain ID.")
    p.add_argument("--dream3d", type=Path, required=True)
    p.add_argument("--grain-id", type=int, required=True)
    p.add_argument("--voxel-csv", type=Path, default=None)
    p.add_argument("--hx", type=_parse_range, default=None)
    p.add_argument("--hy", type=_parse_range, default=None)
    p.add_argument("--hz", type=_parse_range, default=None)
    p.add_argument("--unit-extend-ratio", type=int, default=3)
    p.add_argument("--h5-neighborlist-dset", type=str, default=None)
    p.add_argument("--h5-grain-dset", type=str, default=None)
    p.add_argument(
        "--try-carve",
        action="store_true",
        help="Also call carve_points (slow; confirms full extend path)",
    )
    args = p.parse_args()

    if not args.dream3d.exists():
        print(f"ERROR: dream3d not found: {args.dream3d}", file=sys.stderr)
        return 2
    if args.voxel_csv is not None and not args.voxel_csv.exists():
        print(f"ERROR: voxel-csv not found: {args.voxel_csv}", file=sys.stderr)
        return 2

    return diagnose(
        dream3d=args.dream3d,
        grain_id=args.grain_id,
        hx=args.hx,
        hy=args.hy,
        hz=args.hz,
        voxel_csv=args.voxel_csv,
        unit_extend_ratio=args.unit_extend_ratio,
        h5_neighborlist_dset=args.h5_neighborlist_dset,
        h5_grain_dset=args.h5_grain_dset,
        try_carve=args.try_carve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
