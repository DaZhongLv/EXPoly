import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import h5py

# 只有装了 plotly 才跑这个测试
plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go

from expoly.frames import Frame
from expoly.carve import CarveConfig, prepare_carve

PLOT_ENV = "EXPOLY_PLOT"            # 打开开关才会画图
DREAM_ENV = "EXPOLY_DREAM3D_PATH"    # DREAM.3D 路径（可用默认文件名）
RATIO_ENV = "EXPOLY_RATIO"

def _dream_path() -> Path:
    return Path(os.getenv(DREAM_ENV, "Alpoly_elongate.dream3d"))

pytestmark = [
    pytest.mark.skipif(os.getenv(PLOT_ENV) != "1",
                       reason=f"Set {PLOT_ENV}=1 to enable plotting."),
    pytest.mark.skipif(not _dream_path().exists(),
                       reason=f"DREAM.3D not found: set {DREAM_ENV} or place file at project root."),
]

def _pick_valid_grain_id(p: Path) -> int:
    # 读一小块 FeatureIds，找一个 >0 的 grain id
    with h5py.File(p, "r") as h:
        fid = h["DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds"]
        zmax, ymax, xmax = fid.shape[:3]
        block = fid[0:min(16,zmax), 0:min(16,ymax), 0:min(16,xmax), 0]
        valid = np.unique(block)
        valid = valid[valid > 0]
        assert valid.size > 0, "No positive grain id found in sampled block"
        return int(valid.min())

def test_overlap_plot(tmp_path):
    dream = _dream_path()
    gid = _pick_valid_grain_id(dream)

    # 1) 读 frame + 取该 grain 的体素（H 空间网格）
    frame = Frame(dream, prefer_groups=[
        "CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"
    ])
    out_df = frame.from_ID_to_D(gid).copy()
    out_df["ID"] = gid

    # 2) 生成转换后的 FCC 点云（去掉随机，保证可复现）
    ratio = float(os.getenv(RATIO_ENV, "1.5"))
    cfg = CarveConfig(lattice="FCC", ratio=ratio,
                      random_center=False, rng_seed=0, ci_radius=2 ** 0.5)
    fcc_pts = prepare_carve(out_df, frame, cfg)

    # 3) 采样，避免 HTML 太大
    grid = out_df[["HX","HY","HZ"]].rename(columns={"HX":"X","HY":"Y","HZ":"Z"})
    grid_s = grid.sample(min(len(grid), 10000), random_state=0)
    fcc_s  = pd.DataFrame(fcc_pts, columns=["X","Y","Z"]).sample(min(len(fcc_pts), 10000), random_state=0)

    # 4) 叠加绘图并保存为 HTML
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=grid_s["X"], y=grid_s["Y"], z=grid_s["Z"],
        mode="markers", name="Grid (H)",
        marker=dict(size=2, opacity=0.25)
    ))
    fig.add_trace(go.Scatter3d(
        x=fcc_s["X"], y=fcc_s["Y"], z=fcc_s["Z"],
        mode="markers", name="Transformed FCC",
        marker=dict(size=2, opacity=0.7)
    ))
    fig.update_layout(
        title=f"Grain {gid} — H grid vs Transformed FCC (ratio={cfg.ratio})",
        scene_aspectmode="data", margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    out_html = tmp_path / f"carve_overlap_gid{gid}.html"
    fig.write_html(str(out_html))
    assert out_html.exists()
    print(f"[plot saved] {out_html}")
