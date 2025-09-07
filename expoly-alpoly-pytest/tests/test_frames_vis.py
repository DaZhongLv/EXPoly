# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Literal
import warnings

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

try:
    from expoly.frames import Frame
except Exception:  # pragma: no cover
    from frames import Frame  # 若你直接把 frames.py 放在项目根

class GridTooLargeWarning(UserWarning):
    pass

def visualize_grain(frame_or_path: Union[Frame, str, Path],
                    grain_id: int,
                    mode: Literal["scatter", "voxels"] = "scatter",
                    show_margin: bool = False,
                    downsample: int = 1,
                    max_voxels: int = 800_000,
                    figsize=(8, 8),
                    title: Optional[str] = None):
    if isinstance(frame_or_path, Frame):
        fr = frame_or_path
    else:
        fr = Frame(path=str(frame_or_path))

    if show_margin:
        df = fr.find_grain_NN_with_out(grain_id, with_diagonal=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"grain_id={grain_id} not found or empty (with margin).")
        xyz = df[["HX", "HY", "HZ"]].to_numpy(dtype=int)
        margin = df["margin-ID"].to_numpy(dtype=int)
    else:
        df = fr.from_ID_to_D(grain_id)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"grain_id={grain_id} not found or empty.")
        xyz = df[["HX", "HY", "HZ"]].to_numpy(dtype=int)
        margin = None

    if downsample and downsample > 1 and len(xyz) > downsample:
        xyz = xyz[::downsample, :]
        if margin is not None:
            margin = margin[::downsample]

    xs, ys, zs = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    _mode = mode
    if mode == "voxels":
        xmin, ymin, zmin = xs.min(), ys.min(), zs.min()
        xmax, ymax, zmax = xs.max(), ys.max(), zs.max()
        nx, ny, nz = (xmax - xmin + 1), (ymax - ymin + 1), (zmax - zmin + 1)
        volume = int(nx) * int(ny) * int(nz)
        if volume > max_voxels:
            warnings.warn(
                f"The voxel grid {nx}x{ny}x{nz}={volume:,} exceeds max_voxels={max_voxels:,}. "
                f"Falling back to scatter.", GridTooLargeWarning)
            _mode = "scatter"

    if _mode == "scatter":
        if margin is None:
            ax.scatter(xs, ys, zs, s=1, alpha=0.6)
        else:
            for mid in (0, 1, 2):
                msk = (margin == mid)
                if np.any(msk):
                    ax.scatter(xs[msk], ys[msk], zs[msk], s=2, alpha=0.75, label=f"margin={mid}")
            ax.legend(loc="best", fontsize=8)
    else:
        xmin, ymin, zmin = xs.min(), ys.min(), zs.min()
        xmax, ymax, zmax = xs.max(), ys.max(), zs.max()
        nx, ny, nz = (xmax - xmin + 1), (ymax - ymin + 1), (zmax - zmin + 1)
        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid[(xs - xmin, ys - ymin, zs - zmin)] = True
        ax.voxels(grid, edgecolor=None)

    ax.set_xlabel("HX"); ax.set_ylabel("HY"); ax.set_zlabel("HZ")

    def _set_equal_3d(ax_, X, Y, Z):
        xr = X.max() - X.min()
        yr = Y.max() - Y.min()
        zr = Z.max() - Z.min()
        max_range = max(xr, yr, zr) or 1.0
        cx = (X.max() + X.min()) / 2
        cy = (Y.max() + Y.min()) / 2
        cz = (Z.max() + Z.min()) / 2
        ax_.set_box_aspect((1, 1, 1))
        ax_.set_xlim(cx - max_range / 2, cx + max_range / 2)
        ax_.set_ylim(cy - max_range / 2, cy + max_range / 2)
        ax_.set_zlim(cz - max_range / 2, cz + max_range / 2)

    _set_equal_3d(ax, xs, ys, zs)
    ax.set_title(title or f"Grain {grain_id} ({_mode})")
    plt.tight_layout()
    return fig, ax, len(xs)

@pytest.mark.filterwarnings("ignore:.*Matplotlib.*")
def test_visualize_one_grain(frame_path, grain_id, vis_mode, artifacts_dir,
                             downsample, show_margin):
    fr = Frame(path=str(frame_path))

    for m in vis_mode:
        try:
            fig, ax, npts = visualize_grain(
                fr, grain_id,
                mode=m,
                show_margin=show_margin,
                downsample=downsample,
                max_voxels=800_000,
                figsize=(8, 8),
            )
        except ValueError as e:
            pytest.skip(f"{e} 请选择有效的 --grain-id。")

        out = artifacts_dir / f"grain_{grain_id}_{m}.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)

        print(f"[SAVED] {out}  (points={npts})")  # 关键：总会打印保存路径
        assert out.exists(), f"Expected image not found: {out}"
        assert out.stat().st_size > 0, f"Generated image is empty: {out}"
        assert npts > 0, "No points drawn."

def test_smoke_print_summary(frame_path, grain_id, artifacts_dir):
    print(f"[INFO] frame: {Path(frame_path).resolve()}")
    print(f"[INFO] grain_id: {grain_id}")
    print(f"[INFO] artifacts_dir: {artifacts_dir}")
