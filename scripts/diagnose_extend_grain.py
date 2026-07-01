#!/usr/bin/env python3
"""
Diagnose extend pipeline for a single grain ID.

Uses the same Frame construction as ``expoly run`` (including auto-detected
CSV column names: voxel-X/Y/Z or HX/HY/HZ, grain-ID or grain_id).

Usage (on supercomputer, from repo root):

  python scripts/diagnose_extend_grain.py \\
    --dream3d /path/to/file.dream3d \\
    --voxel-csv /path/to/voxel.csv \\
    --grain-id 1234 \\
    --unit-extend-ratio 3

List grains that expoly run would actually process:

  python scripts/diagnose_extend_grain.py \\
    --dream3d /path/to/file.dream3d \\
    --voxel-csv /path/to/voxel.csv \\
    --list-grains 30

Important: with ``--voxel-csv``, ``--hx/--hy/--hz`` are **0-based local grid
indices** inside the CSV (usually 0:129), **not** Dream3D absolute coords from
the filename (e.g. hx70-200). Omit them to use the full CSV grid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Always prefer this repo's src/ over an older installed expoly package.
_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

from expoly.carve import CarveConfig, carve_points  # noqa: E402
from expoly.cli import (  # noqa: E402
    _build_frame_for_carve,
    _parse_range,
    _pick_grain_ids,
    _resolve_run_h_ranges,
)
from expoly.frames import (  # noqa: E402
    Frame,
    VoxelCSVSchema,
    detect_voxel_csv_columns,
    normalize_axis_to_indices,
    read_voxel_csv_columns,
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


def _print_grain_id_help(gids: np.ndarray, n: int = 20) -> None:
    gids_sorted = np.sort(gids.astype(int))
    print(f"\n--- grains expoly run would process ({len(gids_sorted)} total) ---")
    print(f"  min ID: {gids_sorted[0]}   max ID: {gids_sorted[-1]}")
    head = ", ".join(str(int(g)) for g in gids_sorted[:n])
    tail = ", ".join(str(int(g)) for g in gids_sorted[-min(n, len(gids_sorted)) :])
    print(f"  first {min(n, len(gids_sorted))}: {head}")
    if len(gids_sorted) > n:
        print(f"  last {min(n, len(gids_sorted))}: {tail}")
    print("  Use one of these IDs for --grain-id (real grain ID, not row index).")


def _csv_grain_row_count(csv_path: Path, grain_id: int, grain_col: str) -> int:
    count = 0
    for chunk in pd.read_csv(
        csv_path,
        sep=r"\s+",
        comment="#",
        engine="python",
        usecols=[grain_col],
        chunksize=500_000,
    ):
        count += int((chunk[grain_col].astype(int) == grain_id).sum())
    return count


def _csv_unique_grain_ids(csv_path: Path, grain_col: str, limit: int = 20) -> tuple[int, int, int, list[int]]:
    seen: set[int] = set()
    gmin = 10**18
    gmax = -1
    for chunk in pd.read_csv(
        csv_path,
        sep=r"\s+",
        comment="#",
        engine="python",
        usecols=[grain_col],
        chunksize=500_000,
    ):
        vals = chunk[grain_col].astype(int)
        gmin = min(gmin, int(vals.min()))
        gmax = max(gmax, int(vals.max()))
        seen.update(int(v) for v in vals.unique())
    sample = sorted(g for g in seen if g > 0)[:limit]
    return gmin, gmax, len(seen), sample


def _csv_grain_unique_cells(
    csv_path: Path,
    grain_id: int,
    schema: VoxelCSVSchema,
) -> tuple[int, int]:
    """Return (csv_rows, unique normalized grid cells) for one grain."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    rows = 0
    usecols = [schema.grain_col, schema.x_col, schema.y_col, schema.z_col]
    for chunk in pd.read_csv(
        csv_path,
        sep=r"\s+",
        comment="#",
        engine="python",
        usecols=usecols,
        chunksize=500_000,
    ):
        mask = chunk[schema.grain_col].astype(int) == grain_id
        if not mask.any():
            continue
        sub = chunk.loc[mask]
        rows += len(sub)
        xs.append(sub[schema.x_col].to_numpy())
        ys.append(sub[schema.y_col].to_numpy())
        zs.append(sub[schema.z_col].to_numpy())
    if rows == 0:
        return 0, 0
    hx, _, _ = normalize_axis_to_indices(np.concatenate(xs))
    hy, _, _ = normalize_axis_to_indices(np.concatenate(ys))
    hz, _, _ = normalize_axis_to_indices(np.concatenate(zs))
    cells = set(zip(hz.tolist(), hy.tolist(), hx.tolist()))
    return rows, len(cells)


def _dream3d_grain_size(dream3d: Path, grain_id: int, h5_grain_dset: str | None) -> int | None:
    mapping = {
        "GrainId": h5_grain_dset or "FeatureIds",
        "Euler": "EulerAngles",
        "Num_NN": "NumNeighbors",
        "Num_list": "NeighborList",
        "Dimension": "DIMENSIONS",
    }
    try:
        frame = Frame(str(dream3d), mapping=mapping)
        return int(frame.get_grain_size(grain_id))
    except Exception as exc:
        print(f"  (full Dream3D check skipped: {exc})")
        return None


def _resolve_schema(voxel_csv: Path | None, grain_col: str | None) -> VoxelCSVSchema | None:
    if voxel_csv is None:
        return None
    schema = detect_voxel_csv_columns(read_voxel_csv_columns(voxel_csv))
    if grain_col is not None:
        schema = VoxelCSVSchema(schema.x_col, schema.y_col, schema.z_col, grain_col)
    return schema


def _crosscheck_csv(
    voxel_csv: Path,
    schema: VoxelCSVSchema,
    grain_id: int,
    in_gids: bool,
    frame_voxel_count: int,
    dream3d: Path,
    h5_grain_dset: str | None,
) -> None:
    print("\n--- CSV cross-check (same column mapping as VoxelCSVFrame) ---")
    print(f"  CSV columns detected: x={schema.x_col!r} y={schema.y_col!r} "
          f"z={schema.z_col!r} grain={schema.grain_col!r}")
    gmin, gmax, n_grains, sample = _csv_unique_grain_ids(voxel_csv, schema.grain_col)
    print(f"  unique grain IDs in CSV file: {n_grains}  range [{gmin}, {gmax}]")
    print(f"  sample IDs: {sample}")

    csv_rows = _csv_grain_row_count(voxel_csv, grain_id, schema.grain_col)
    csv_cells = 0
    if csv_rows > 0:
        csv_rows, csv_cells = _csv_grain_unique_cells(voxel_csv, grain_id, schema)
    print(f"  CSV rows with {schema.grain_col}=={grain_id}: {csv_rows}")
    if csv_rows > 0:
        print(f"  unique normalized grid cells for that grain in CSV: {csv_cells}")
    print(f"  voxels in loaded Frame grid (get_grain_size): {frame_voxel_count}")

    if csv_rows > 0 and frame_voxel_count == 0:
        print(
            "\n  >>> CSV has rows but Frame grid is empty for this grain."
        )
        if csv_cells < csv_rows:
            print(
                f"  >>> {csv_rows - csv_cells} duplicate (z,y,x) cells in CSV;"
                " last row wins when building the grid — some IDs may disappear."
            )
        print(
            "  >>> Check that --grain-col / coordinate columns match your CSV header."
            " Re-run with detected columns printed above."
        )
    elif csv_rows == 0:
        print(f"\n  >>> grain {grain_id} has zero rows in the CSV file.")
        full_n = _dream3d_grain_size(dream3d, grain_id, h5_grain_dset)
        if full_n is not None:
            print(f"  >>> full .dream3d volume voxels for this grain: {full_n}")
            if full_n > 0:
                print(
                    "  >>> Grain exists in Dream3D but not in this voxel CSV crop."
                    " Use a CSV that includes it, or omit --voxel-csv and pass Dream3D --hx/--hy/--hz."
                )
    elif csv_rows > 0 and in_gids and frame_voxel_count > 0:
        print("\n  CSV and Frame agree: this grain is present.")


def diagnose(
    dream3d: Path,
    grain_id: int | None,
    hx: tuple[int, int] | None,
    hy: tuple[int, int] | None,
    hz: tuple[int, int] | None,
    voxel_csv: Path | None,
    unit_extend_ratio: int,
    h5_neighborlist_dset: str | None,
    h5_grain_dset: str | None,
    grain_col: str | None,
    list_grains: int | None,
    try_carve: bool,
) -> int:
    schema = _resolve_schema(voxel_csv, grain_col)
    hx_r, hy_r, hz_r = _resolve_run_h_ranges(voxel_csv, hx, hy, hz)

    print(f"dream3d: {dream3d}")
    if voxel_csv:
        print(f"voxel-csv: {voxel_csv}")
    if schema is not None:
        print(
            "CSV schema: "
            f"x={schema.x_col} y={schema.y_col} z={schema.z_col} grain={schema.grain_col}"
        )
    print(f"H ranges (local grid indices): HX={hx_r} HY={hy_r} HZ={hz_r}")
    if voxel_csv is not None and any(v is not None for v in (hx, hy, hz)):
        print(
            "NOTE: with --voxel-csv, --hx/--hy/--hz are local 0-based indices inside the CSV,"
            " not Dream3D absolute coords from the filename."
        )
    if grain_id is not None:
        print(f"grain-id: {grain_id}")
    print(f"unit-extend-ratio: {unit_extend_ratio}")
    print(f"h5-grain-dset: {h5_grain_dset or 'FeatureIds (auto-detect GrainID in Frame)'}")

    frame = _build_frame_for_carve(
        dream3d,
        voxel_csv,
        h5_grain_dset=h5_grain_dset,
        h5_neighborlist_dset=h5_neighborlist_dset,
    )
    print(f"volume limits: HX=[0,{frame.HX_lim}) HY=[0,{frame.HY_lim}) HZ=[0,{frame.HZ_lim})")
    print(f"frame type: {type(frame).__name__}")

    gids = _pick_grain_ids(frame, hx_r, hy_r, hz_r)
    _print_grain_id_help(gids, n=list_grains or 20)

    if list_grains is not None and grain_id is None:
        return 0

    if grain_id is None:
        print("\nERROR: pass --grain-id <ID> and/or --list-grains N.", file=sys.stderr)
        return 2

    in_gids = bool(np.any(gids == grain_id))
    frame_voxel_count = int(frame.get_grain_size(grain_id))
    if not in_gids:
        print(f"\nWARNING: grain {grain_id} is NOT in the selected H-range grain list.")
    else:
        idx = int(np.where(gids == grain_id)[0][0]) + 1
        print(f"  grain {grain_id} is #{idx} in sorted unique gids (of {len(gids)})")

    if voxel_csv is not None and schema is not None:
        _crosscheck_csv(
            voxel_csv, schema, grain_id, in_gids, frame_voxel_count, dream3d, h5_grain_dset
        )

    out = frame.from_ID_to_D(grain_id)
    _summary("0) from_ID_to_D (grain voxels)", out)

    if in_gids and len(out) == 0:
        print(
            "\n>>> INTERNAL INCONSISTENCY: grain is in H-range list but from_ID_to_D is empty."
            " Please paste the full script output (including CSV schema lines)."
        )
        return 1

    if len(out) == 0:
        print(
            "\n>>> STOP: no voxels for this grain in the loaded volume."
            " Use --list-grains and pick an ID from that list."
        )
        return 1

    out_margin = frame.find_grain_NN_with_out(grain_id)
    _summary("1) find_grain_NN_with_out", out_margin)

    if len(out_margin) == 0:
        print("\n>>> STOP: out_margin is empty — extend will fail here.")
        return 1

    extend_raw = frame.get_extend_Out_(out_margin, unit_extend_ratio)
    _summary("2) get_extend_Out_", extend_raw)

    extend = frame.renew_outer_margin(extend_raw)
    _summary("3) renew_outer_margin", extend)

    extend_xyz = extend.rename(columns={"grain-ID": "ID"}).copy()
    _summary("4) extend_xyz (input to carve_points)", extend_xyz)

    if len(extend_xyz) == 0:
        print("\n>>> FAIL: extend_xyz is EMPTY — expoly run --extend will crash with:")
        print("    IndexError: single positional indexer is out-of-bounds")
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
    p.add_argument("--grain-id", type=int, default=None, help="Real grain ID from CSV / Dream3D")
    p.add_argument("--voxel-csv", type=Path, default=None)
    p.add_argument(
        "--hx",
        type=_parse_range,
        default=None,
        help="Local X index range inside CSV grid (NOT Dream3D absolute coords)",
    )
    p.add_argument("--hy", type=_parse_range, default=None, help="Local Y index range inside CSV")
    p.add_argument("--hz", type=_parse_range, default=None, help="Local Z index range inside CSV")
    p.add_argument("--unit-extend-ratio", type=int, default=3)
    p.add_argument("--h5-neighborlist-dset", type=str, default=None)
    p.add_argument(
        "--h5-grain-dset",
        type=str,
        default=None,
        help="HDF5 grain dataset (Meshed files: try GrainID)",
    )
    p.add_argument("--grain-col", type=str, default=None, help="Override CSV grain column name")
    p.add_argument("--list-grains", type=int, default=None, metavar="N")
    p.add_argument("--try-carve", action="store_true")
    args = p.parse_args()

    if args.grain_id is None and args.list_grains is None:
        p.error("pass --grain-id <ID> and/or --list-grains N")

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
        grain_col=args.grain_col,
        list_grains=args.list_grains,
        try_carve=args.try_carve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
