# Blender 渲染脚本使用指南

## 功能

`blender_render.py` 是一个 Blender Python 脚本，用于：
1. 加载点云文件（PLY 格式）
2. 使用 Geometry Nodes 在点上实例化立方体
3. 为 grain 和 margin 分配不同材质（灰色和淡蓝色，半透明）
4. 渲染旋转动画（61 帧，180°）

## macOS 运行命令

### 基本用法

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_render.py \
    -- \
    --grain-ply grain_111_points.ply \
    --margin-ply grain_111_margin_points.ply
```

### 完整参数示例

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_render.py \
    -- \
    --grain-ply grain_111_points.ply \
    --margin-ply grain_111_margin_points.ply \
    --voxel-size 1.0 \
    --output-dir frames \
    --n-frames 61 \
    --step-deg 3.0 \
    --resolution-x 1920 \
    --resolution-y 1080
```

### 只渲染 grain（无 margin）

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_render.py \
    -- \
    --grain-ply grain_111_points.ply
```

## 参数说明

- `--grain-ply`: grain 点云 PLY 文件路径（必需）
- `--margin-ply`: margin 点云 PLY 文件路径（可选）
- `--voxel-size`: 体素立方体大小（默认: 1.0）
- `--output-dir`: 渲染输出目录（默认: frames）
- `--n-frames`: 总帧数（默认: 61）
- `--step-deg`: 每帧旋转角度（度，默认: 3.0）
- `--resolution-x`: 渲染分辨率 X（默认: 1920）
- `--resolution-y`: 渲染分辨率 Y（默认: 1080）
- `--grain-color`: grain 颜色（hex 格式，默认: #8E8E93）
- `--margin-color`: margin 颜色（hex 格式，默认: #A7C7E7）

## 输出

脚本会在指定的输出目录（默认 `frames/`）生成 PNG 序列：
- `frame_0001.png`
- `frame_0002.png`
- ...
- `frame_0061.png`

## 工作流程

1. **加载点云**：从 PLY 文件读取点云数据
2. **创建 Geometry Nodes**：
   - Mesh to Points：将点云转换为点
   - Instance on Points：在每个点上实例化立方体
   - Realize Instances：将实例转换为实际几何体
3. **分配材质**：
   - Grain：灰色 (#8E8E93)，透明度 0.4
   - Margin：淡蓝色 (#A7C7E7)，透明度 0.17
4. **设置相机**：固定在 XY 平面，viewup=(0,0,1)
5. **动画**：对象绕 Z 轴旋转，每帧 3°，共 61 帧（180°）
6. **渲染**：使用 Cycles 引擎渲染 PNG 序列

## 注意事项

- 脚本在后台运行（`--background`），不会打开 Blender GUI
- 确保 PLY 文件路径正确
- 渲染时间取决于点云大小和分辨率
- 可以使用 `--resolution-x` 和 `--resolution-y` 调整输出质量

## 故障排除

### 找不到 Blender

如果 Blender 不在默认位置，使用完整路径：

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python blender_render.py -- ...
```

### PLY 文件无法加载

确保 PLY 文件格式正确，可以使用 `export_grain_points.py` 生成。

### 渲染太慢

- 减少 `--n-frames`（例如 31 帧）
- 降低分辨率（例如 1280x720）
- 在脚本中减少 Cycles samples（当前为 128）
