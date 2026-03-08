# EXPoly pipeline visualization (web app)

Left: experimental data (grain = interior + inner shell; margin = outer shell only, 1 layer). Right: simulation in 5 steps. Both 3D views share the same camera. Grain selector and **Load** button at the bottom of the left panel for real-time grain switch (after first export).

## 1. Generate precomputed data

From the project root (requires a Dream3D HDF5 path):

```bash
python scripts/export_legacy_steps_for_web.py \
  --dream3d /path/to/your.dream3d \
  --grain-id 100 \
  --ratio 1.5 \
  --out-dir web_app/data
```

Default grain ID is 100. If your HDF5 uses different dataset names (e.g. An0new6.dream3d uses NeighborList2), pass the same `--h5-*-dset` options as for `expoly run`:

```bash
python scripts/export_legacy_steps_for_web.py \
  --dream3d An0new6.dream3d \
  --h5-grain-dset FeatureIds \
  --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors \
  --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --out-dir web_app/data
```

Or with toy data (generate first):

```bash
python examples/toy_data_generator.py --output examples/toy_data.dream3d
python scripts/export_legacy_steps_for_web.py --dream3d examples/toy_data.dream3d --out-dir web_app/data
```

Outputs go to `web_app/data/`: `experimental.parquet`, `step1_ball.parquet` … `step5_craved_gb.parquet`, `grain_id.txt`, `dream3d_path.txt` (5 steps; default grain ID is 100).

## 2. Install and run

```bash
pip install "expoly[web]"
# or in dev mode: pip install -e ".[web]"

# Dash only (Load button still works if export was run once)
python web_app/app.py

# With API for real-time grain: POST /api/export (from project root)
python -m uvicorn web_app.server:app --port 8050
```

Open http://127.0.0.1:8050 . Left: experimental 3D, then grain ID + **Load** at bottom; right: simulation 3D (same height) and 5-step selector below. Enter a grain ID and click **Load** to refresh data without re-running the script. Rotating one view rotates the other.

## 2.1 清理占用端口（Port 8050 in use）

若出现 `Port 8050 in use; using port 8051` 等，说明 8050 已被占用。可先关掉占用该端口的进程：

**macOS / Linux：**

```bash
# 查看占用 8050 的进程
lsof -i :8050

# 结束占用 8050 的进程（可选 -9 强制）
kill $(lsof -ti :8050)
# 若需强制： kill -9 $(lsof -ti :8050)
```

若 8051、8053 等也被占用，可一并清理：

```bash
for p in 8050 8051 8052 8053; do kill $(lsof -ti :$p) 2>/dev/null; done
```

## 3. Margin 对应关系（实验 vs 模拟第五步）

Frame 里对每个晶粒会算出一套 **margin-ID**（见 `find_grain_NN_with_out`）：

- **margin-ID 0**：晶粒内部（不在边界上）
- **margin-ID 1**：**外壳** —— 晶粒外、紧贴晶粒的那一层 voxel（6 邻接扩展出去的一层）
- **margin-ID 2**：**内壳** —— 晶粒内、贴在界面上的 voxel（与外壳相邻的晶粒侧）

**实验视图（左侧）** 三区着色（与第五步配色统一）：

- **grain** = margin-ID 0（晶粒内部），颜色 royalblue
- **outer margin (1 layer)** = margin-ID 1（**外壳**，晶粒外一层），颜色 gold
- **inner margin** = margin-ID 2（**内壳**，晶粒侧边界），颜色 coral

**模拟第五步（右侧）** 用的是 `step5_craved_gb.parquet`（`carve_gb_m1` 只保留 margin-ID 0 和 2，**没有 margin-ID 1**）：

- **grain** = margin-ID 0，颜色 royalblue
- **inner margin** = margin-ID 2，颜色 coral

第五步提供可选 **“Overlay experimental outer margin (ID 1) by coordinates”**：勾选后会在模拟图上按坐标叠加显示实验的外壳（margin-ID 1）voxel 位置，便于与内壳对比。

结论：**实验里的 margin** 和 **模拟第五步的 margin** 不是同一批 voxel：

- 实验 margin = **外壳**（margin-ID 1，晶粒外一层）
- 模拟第五步 margin = **内壳**（margin-ID 2，晶粒内边界）

它们是**同一条界面的两侧**：实验画的是界面外侧一层，模拟第五步画的是界面内侧（晶粒侧）边界。
