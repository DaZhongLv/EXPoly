#!/usr/bin/env python3
"""
Export legacy pipeline steps to parquet for the web visualization app.
Replicates the logic from legacy_scripts/grid.py and readDATA.py without modifying them.
Uses expoly.frames.Frame and expoly.general_func for Dream3D and Euler/rotation.
Output: web_app/data/experimental.parquet, step1_ball.parquet, ..., step6_craved_gb.parquet.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import spatial

# Add project root so we can import expoly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expoly.frames import Frame
from expoly.general_func import eul2rot_bunge


def unit_vec_ratio(out_xyz: np.ndarray, unit_offsets: np.ndarray, ratio: float) -> np.ndarray:
    """Same as legacy grid.Unit_vec_ratio / SC2FCC base."""
    out_xyz = np.asarray(out_xyz, dtype=float).reshape(-1, 3)
    unit_offsets = np.asarray(unit_offsets, dtype=float).reshape(-1, 3)
    rep = np.repeat(out_xyz, len(unit_offsets), axis=0)
    til = np.tile(unit_offsets, (len(out_xyz), 1)) * float(ratio)
    return rep + til


def sc2fcc(data: np.ndarray, ratio: float) -> np.ndarray:
    """Legacy grid.SC2FCC."""
    unit_fcc = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return unit_vec_ratio(data, unit_fcc, ratio)


def move_to_hcenter(data: np.ndarray, hcenter: list) -> np.ndarray:
    """Legacy grid.move_to_Hcenter."""
    data = np.asarray(data, dtype=float).reshape(-1, 3)
    data[:, 0] += hcenter[0]
    data[:, 1] += hcenter[1]
    data[:, 2] += hcenter[2]
    return data


def prepare_carve_fcc_intermediates(
    frame: Frame,
    Out: pd.DataFrame,
    ratio: float,
    random: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, float, np.ndarray]:
    """
    Replicate legacy prepare_carve_FCC and return (ball_struct, FCC_struct, Ro_FCC, Tr_FCC, R, hcenter, crave_r, avg_eul).
    Out must have columns HX, HY, HZ, ID (or HZ, HY, HX, ID).
    """
    grain_id = int(Out["ID"].iloc[0])
    avg_eul = frame.search_avg_Euler(grain_id)
    R = eul2rot_bunge(avg_eul)

    hx_min, hx_max = Out["HX"].min(), Out["HX"].max()
    hy_min, hy_max = Out["HY"].min(), Out["HY"].max()
    hz_min, hz_max = Out["HZ"].min(), Out["HZ"].max()
    hcenter = [
        0.5 * (hx_max + hx_min),
        0.5 * (hy_max + hy_min),
        0.5 * (hz_max + hz_min),
    ]
    crave_r = (math.dist([hx_min, hy_min, hz_min], [hx_max, hy_max, hz_max]) / 2) * 1.5
    one_dim = np.arange(-crave_r, crave_r + 1, ratio, dtype=float)
    cube_struct = [[i, j, k] for i in one_dim for j in one_dim for k in one_dim]
    tree = spatial.cKDTree(cube_struct)
    if random:
        ball_index = tree.query_ball_point(np.random.sample(3) * 2, crave_r, workers=-1)
    else:
        ball_index = tree.query_ball_point([0, 0, 0], crave_r, workers=-1)
    ball_struct = np.take(cube_struct, ball_index, axis=0)
    FCC_struct = sc2fcc(ball_struct, ratio)
    Ro_FCC = np.dot(FCC_struct, R.T)
    Tr_FCC = move_to_hcenter(Ro_FCC, hcenter)
    return ball_struct, FCC_struct, Ro_FCC, Tr_FCC, R, hcenter, crave_r, avg_eul


def prepare_carve_fcc_tr(frame: Frame, Out: pd.DataFrame, ratio: float, random: bool = False) -> np.ndarray:
    """Return only Tr_FCC (for carve_FCC)."""
    _, _, _, Tr_FCC, *_ = prepare_carve_fcc_intermediates(frame, Out, ratio, random)  # 8-tuple, Tr_FCC is 4th
    return Tr_FCC


def carve_fcc(
    frame: Frame,
    Out: pd.DataFrame,
    ratio: float,
    ci_r: float = np.sqrt(2),
) -> np.ndarray:
    """Legacy carve_FCC: experimental voxel Out + rotated FCC ball -> Crave_FCC (FCC points near voxels)."""
    Tr_FCC = prepare_carve_fcc_tr(frame, Out, ratio, random=False)
    tree_H = spatial.cKDTree(Out[["HX", "HY", "HZ"]].to_numpy())
    tree_C = spatial.cKDTree(Tr_FCC)
    crave_idx = tree_H.query_ball_tree(tree_C, r=ci_r)
    crave_idx_unique = np.unique([item for items in crave_idx for item in items])
    return np.take(Tr_FCC, crave_idx_unique, axis=0)


def carve_gb_m1(Margin: pd.DataFrame, FCC_data: np.ndarray) -> pd.DataFrame:
    """Legacy carve_GB_M1: keep FCC points that fall in margin-ID 0 or 2."""
    Margin_sub = Margin[["HX", "HY", "HZ", "margin-ID"]].copy()
    fcc_h = np.rint(FCC_data).astype(int)
    FCC_df = pd.DataFrame(np.hstack((FCC_data, fcc_h)), columns=["X", "Y", "Z", "HX", "HY", "HZ"])
    merged = FCC_df.merge(Margin_sub, on=["HX", "HY", "HZ"], how="left")
    keep = merged[(merged["margin-ID"] == 2) | (merged["margin-ID"] == 0)]
    return keep[["X", "Y", "Z", "HX", "HY", "HZ", "margin-ID"]].copy()


def process_pre(frame: Frame, grain_id: int, cube_ratio: float) -> np.ndarray:
    """Legacy Process_pre: single grain carve."""
    Out = frame.from_ID_to_D(grain_id)
    return carve_fcc(frame, Out, cube_ratio, ci_r=np.sqrt(2))


def process(frame: Frame, grain_id: int, cube_ratio: float) -> pd.DataFrame:
    """Legacy Process: Margin + carve_FCC + carve_GB_M1."""
    Margin = frame.find_grain_NN_with_out(grain_id)
    Craved_FCC = process_pre(frame, grain_id, cube_ratio)
    Craved_GB = carve_gb_m1(Margin, Craved_FCC)
    Craved_GB["grain-ID"] = grain_id
    return Craved_GB


def process_extend(
    frame: Frame,
    grain_id: int,
    cube_ratio: float,
    unit_extend_ratio: int = 3,
) -> pd.DataFrame:
    """Legacy Process_Extend: extended region (two grains) + carve + GB keep."""
    Out_ = frame.find_grain_NN_with_out(grain_id)
    Extend_Out_ = frame.get_extend_Out_(Out_, unit_extend_ratio)
    Extend_Out_ = frame.renew_outer_margin(Extend_Out_)
    Extend_Out_XYZdf = Extend_Out_.copy()
    Extend_Out_XYZdf = Extend_Out_XYZdf.rename(columns={"grain-ID": "ID"})
    Craved_FCC_ = carve_fcc(frame, Extend_Out_XYZdf, cube_ratio, ci_r=np.sqrt(2))
    Craved_GB = carve_gb_m1(Extend_Out_, Craved_FCC_)
    Craved_GB["grain-ID"] = grain_id
    return Craved_GB, Extend_Out_


def run_export(
    dream3d_path: Path,
    grain_id: int,
    out_dir: Path,
    mapping: dict | None = None,
    ratio: float = 1.5,
) -> None:
    """
    Run the legacy pipeline for one grain and write parquet files to out_dir.
    mapping: HDF5 dataset names, e.g. {"GrainId": "FeatureIds", "Num_list": "NeighborList2", ...}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mapping is None:
        mapping = {
            "GrainId": "FeatureIds",
            "Euler": "EulerAngles",
            "Num_NN": "NumNeighbors",
            "Num_list": "NeighborList",
            "Dimension": "DIMENSIONS",
        }
    frame = Frame(dream3d_path, mapping=mapping)
    Out = frame.from_ID_to_D(grain_id)

    ball_struct, FCC_struct, Ro_FCC, Tr_FCC, R, hcenter, crave_r, avg_eul = prepare_carve_fcc_intermediates(
        frame, Out, ratio, random=False
    )
    pd.DataFrame(ball_struct, columns=["X", "Y", "Z"]).to_parquet(
        out_dir / "step1_ball.parquet", index=False
    )
    pd.DataFrame(FCC_struct, columns=["X", "Y", "Z"]).to_parquet(
        out_dir / "step2_fcc.parquet", index=False
    )
    pd.DataFrame(Tr_FCC, columns=["X", "Y", "Z"]).to_parquet(
        out_dir / "step3_rotated.parquet", index=False
    )
    # Step 3 transform for axes/planes in app (Euler radians so app can recompute R, plus R, hcenter, ball radius)
    step3_transform = {
        "euler": np.asarray(avg_eul, dtype=float).tolist(),
        "R": R.tolist(),
        "hcenter": hcenter,
        "ball_radius": float(crave_r),
    }
    (out_dir / "step3_transform.json").write_text(json.dumps(step3_transform))

    Crave_FCC = carve_fcc(frame, Out, ratio, ci_r=np.sqrt(2))
    pd.DataFrame(Crave_FCC, columns=["X", "Y", "Z"]).to_parquet(
        out_dir / "step4_carved.parquet", index=False
    )
    Margin = frame.find_grain_NN_with_out(grain_id)
    # Step 4 mesh: grain + inner margin only (margin-ID 0 and 2)
    voxel_mesh = Margin[Margin["margin-ID"].isin([0, 2])][["HZ", "HY", "HX"]].copy()
    voxel_mesh.to_parquet(out_dir / "step4_voxel.parquet", index=False)

    Craved_GB = carve_gb_m1(Margin, Crave_FCC)
    Craved_GB["grain-ID"] = grain_id
    Margin.to_parquet(out_dir / "step5_margin.parquet", index=False)
    Craved_GB.to_parquet(out_dir / "step5_craved_gb.parquet", index=False)

    Margin.to_parquet(out_dir / "experimental.parquet", index=False)
    (out_dir / "grain_id.txt").write_text(str(grain_id))
    (out_dir / "dream3d_path.txt").write_text(str(Path(dream3d_path).resolve()))

    # Scale (CR) viz: precompute fine-carved grain for CR 1.0, 1.5, 2.0 (no extend + extend=3)
    SCALE_RATIOS = [1.0, 1.5, 2.0]
    for cr in SCALE_RATIOS:
        Craved_GB_cr = process(frame, grain_id, cr)
        Craved_GB_cr.to_parquet(out_dir / f"scale_cr{cr}.parquet", index=False)
        Craved_GB_ext, _ = process_extend(frame, grain_id, cr, unit_extend_ratio=3)
        Craved_GB_ext.to_parquet(out_dir / f"scale_extend3_cr{cr}.parquet", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export legacy pipeline steps for web viz")
    parser.add_argument("--dream3d", type=Path, required=True, help="Path to Dream3D HDF5 file")
    parser.add_argument("--grain-id", type=int, default=100, help="Grain ID to use (default 100)")
    parser.add_argument("--ratio", type=float, default=1.5, help="Lattice ratio (default 1.5)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "web_app" / "data",
        help="Output directory for parquet files",
    )
    parser.add_argument("--h5-grain-dset", type=str, default=None, help="HDF5 grain ID dataset (default: FeatureIds)")
    parser.add_argument("--h5-euler-dset", type=str, default=None, help="HDF5 Euler angles dataset (default: EulerAngles)")
    parser.add_argument("--h5-numneighbors-dset", type=str, default=None, help="HDF5 NumNeighbors dataset (default: NumNeighbors)")
    parser.add_argument("--h5-neighborlist-dset", type=str, default=None, help="HDF5 NeighborList dataset (default: NeighborList; e.g. NeighborList2 for An0new6.dream3d)")
    parser.add_argument("--h5-dimensions-dset", type=str, default=None, help="HDF5 dimensions dataset (default: DIMENSIONS)")
    args = parser.parse_args()

    mapping = {
        "GrainId": args.h5_grain_dset or "FeatureIds",
        "Euler": args.h5_euler_dset or "EulerAngles",
        "Num_NN": args.h5_numneighbors_dset or "NumNeighbors",
        "Num_list": args.h5_neighborlist_dset or "NeighborList",
        "Dimension": args.h5_dimensions_dset or "DIMENSIONS",
    }
    print(f"Using grain ID: {args.grain_id}")
    run_export(args.dream3d, args.grain_id, args.out_dir, mapping, args.ratio)
    print("Exported to", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
