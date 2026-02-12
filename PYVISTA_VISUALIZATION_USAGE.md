# PyVista 3D 可视化脚本使用指南

## 安装依赖

```bash
pip install pyvista pillow scipy numpy h5py
```

## 基本用法

### 1. 查看 HDF5 文件结构

如果不知道数据路径，先查看文件结构：

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --print-h5-tree
```

### 2. 自动检测路径（推荐）

脚本会自动检测常见的数据路径：

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111
```

### 3. 显式指定路径

如果自动检测失败，可以显式指定：

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --grain-id-path "DataContainers/ImageDataContainer/CellData/FeatureIds" \
    --positions-path "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/DIMENSIONS"
```

注意：如果使用 `--positions-path` 指向 DIMENSIONS，脚本会自动从维度生成网格坐标。

### 4. 自定义参数

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --cube-size 1.2 \
    --margin-dist 2.5 \
    --step-deg 3.0 \
    --n-frames 61 \
    --fps 10 \
    --output-dir my_visualizations
```

### 5. 只渲染特定 Stage

```bash
# 只渲染 Stage A (grain only)
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --stage A

# 只渲染 Stage B (grain + margin)
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --stage B
```

## 参数说明

### 输入参数

- `--dream3d`: HDF5 文件路径（必需）
- `--grain-id`: 目标 grain ID（必需）
- `--print-h5-tree`: 打印 HDF5 文件结构并退出
- `--positions-path`: 显式指定位置数据路径
- `--grain-id-path`: 显式指定 grain ID 数据路径
- `--no-auto-detect`: 禁用自动检测

### 可视化参数

- `--cube-size`: 每个体素立方体的尺寸（默认: 1.0）
- `--margin-dist`: margin 距离阈值（默认: 2.0）

### 动画参数

- `--step-deg`: 每帧旋转角度（度，默认: 3.0）
- `--n-frames`: 总帧数（默认: 61，对应 180°）
- `--fps`: GIF 帧率（默认: 10）

### 相机参数

- `--camera-x`: 相机 X 位置（自动计算如果未指定）
- `--camera-y`: 相机 Y 位置（自动计算如果未指定）
- `--camera-z`: 相机 Z 位置（自动计算如果未指定）

### 输出参数

- `--output-dir`: 输出目录（默认: visualizations_pyvista）
- `--stage`: 渲染阶段：A（仅 grain）、B（grain+margin）、both（两者，默认）

## 输出文件

脚本会在输出目录下创建：

```
visualizations_pyvista/
├── stage_A/
│   ├── stage_A_frame_0000.png
│   ├── stage_A_frame_0001.png
│   ├── ...
│   └── stage_A_turntable.gif
└── stage_B/
    ├── stage_B_frame_0000.png
    ├── stage_B_frame_0001.png
    ├── ...
    └── stage_B_turntable.gif
```

## 完整示例

### 示例 1: 使用 An0new6.dream3d（Dream3D 格式）

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --grain-id-path "DataContainers/ImageDataContainer/CellData/FeatureIds" \
    --positions-path "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/DIMENSIONS" \
    --cube-size 1.0 \
    --margin-dist 2.0 \
    --n-frames 61 \
    --fps 10 \
    --stage both
```

### 示例 2: 高质量渲染（更多帧）

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --step-deg 2.0 \
    --n-frames 91 \
    --fps 15 \
    --cube-size 1.2
```

### 示例 3: 快速预览（较少帧）

```bash
python visualize_grain_pyvista.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --step-deg 6.0 \
    --n-frames 31 \
    --fps 8 \
    --stage A
```

## 技术细节

### Stage A: Grain Only
- 颜色: Apple gray (#8E8E93)
- 透明度: 0.4
- 显示边缘: 是（浅灰色 #C0C0C0）

### Stage B: Grain + Margin
- Grain: 同 Stage A
- Margin: 淡蓝色 (#A7C7E7)，透明度 0.17，无边缘

### 相机设置
- 固定在 XY 平面
- viewup = (0, 0, 1)（Z 轴向上）
- focal_point = grain 中心
- 通过旋转对象实现动画（不是旋转相机）

### 固定视野
- 计算 grain + margin 的边界框
- 将范围扩大 2 倍（每个方向 half-extent × 2）
- 使用不可见边界框锁定视野，防止自动缩放

## 故障排除

### 问题 1: 找不到数据集路径

**解决方案：**
```bash
# 先查看文件结构
python visualize_grain_pyvista.py --dream3d An0new6.dream3d --grain-id 111 --print-h5-tree

# 然后使用显式路径
python visualize_grain_pyvista.py --dream3d An0new6.dream3d --grain-id 111 \
    --grain-id-path "实际路径" \
    --positions-path "实际路径"
```

### 问题 2: 内存不足

**解决方案：**
- 减少帧数：`--n-frames 31`
- 使用较小的 cube_size：`--cube-size 0.8`
- 只渲染一个 stage：`--stage A`

### 问题 3: GIF 生成失败

**解决方案：**
```bash
pip install pillow
```

### 问题 4: PyVista 渲染问题

**解决方案：**
```bash
# 确保使用 off-screen 渲染
# 如果仍有问题，检查 OpenGL 支持
python -c "import pyvista as pv; print(pv.Report())"
```

## 性能优化建议

1. **减少帧数**：对于快速预览，使用 `--n-frames 31` 或更少
2. **降低分辨率**：修改脚本中的 `window_size=[1920, 1080]` 为更小的值
3. **只渲染需要的 stage**：使用 `--stage A` 或 `--stage B`
4. **使用更小的 cube_size**：可以减少渲染的几何复杂度
