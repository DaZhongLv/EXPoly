# Blender 工作流程动画使用指南

## 概述

`blender_animation_workflow.py` 脚本用于创建多阶段工作流程动画，展示从网格球到 FCC 晶格，再到与实验数据重叠的完整过程。

## 动画阶段

脚本创建了 5 个连续的动画阶段（Shot 0-4），总共 260 帧：

- **Shot 0 (帧 1-40)**: 网格球与局部坐标系
- **Shot 1 (帧 41-80)**: FCC 晶格生成
- **Shot 2 (帧 81-160)**: FCC 网格球旋转（局部坐标系）
- **Shot 3 (帧 161-220)**: 与实验晶粒点云重叠
- **Shot 4 (帧 221-260)**: 最终重叠结果（仅显示晶粒内部的 FCC 点）

## 使用方法

### 基本用法

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_animation_workflow.py \
    -- \
    --grain-ply grain_111_points.ply
```

### 完整参数示例

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_animation_workflow.py \
    -- \
    --grain-ply grain_111_points.ply \
    --output-blend workflow_animation.blend \
    --point-size 0.5 \
    --grid-ball-radius 20.0 \
    --grid-spacing 1.0 \
    --fcc-lattice-constant 3.524 \
    --resolution-x 1920 \
    --resolution-y 1080 \
    --use-scipy
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grain-ply` | **必需** | 晶粒点云 PLY 文件路径 |
| `--output-blend` | `workflow_animation.blend` | 输出的 Blender 文件路径 |
| `--point-size` | `0.5` | 点云球体大小 |
| `--grid-ball-radius` | `20.0` | 网格球半径 |
| `--grid-spacing` | `1.0` | 网格间距 |
| `--fcc-lattice-constant` | `3.524` | FCC 晶格常数 |
| `--resolution-x` | `1920` | 渲染分辨率 X |
| `--resolution-y` | `1080` | 渲染分辨率 Y |
| `--use-scipy` | `False` | 使用 scipy 加速点过滤（需要安装 scipy） |

## 场景结构

脚本创建以下对象层次结构：

```
RotationPivot (空对象，用于旋转)
├── GridBallPoints (网格球点云)
├── FCCBallPoints (FCC 晶格点云)
├── FCCInsideGrain (晶粒内部的 FCC 点，高亮显示)
├── GrainPointCloud (实验晶粒点云)
└── AxisHelper (XYZ 坐标轴辅助器)
    ├── AxisHelper_X_Shaft (X 轴，红色)
    ├── AxisHelper_X_Arrow (X 箭头)
    ├── AxisHelper_Y_Shaft (Y 轴，绿色)
    ├── AxisHelper_Y_Arrow (Y 箭头)
    ├── AxisHelper_Z_Shaft (Z 轴，蓝色)
    └── AxisHelper_Z_Arrow (Z 箭头)
```

## 材质设置

脚本自动创建以下材质：

- **MAT_GridDefault**: 网格/FCC 默认材质（浅灰色，半透明）
- **MAT_GridHighlight**: 网格/FCC 高亮材质（暖色，橙色/红色，带发光）
- **MAT_Grain**: 晶粒材质（中性浅灰色，半透明）

## 相机设置

- 相机位置固定在整个动画过程中
- 相机位置：`(50, -50, 30)`
- 相机旋转：`(60°, 0°, 45°)`
- 所有旋转都应用于 `RotationPivot` 对象，而不是相机

## 动画关键帧

### Shot 0: 网格球与局部坐标系
- 显示网格球点云
- 显示 XYZ 坐标轴
- 无旋转

### Shot 1: FCC 晶格生成
- FCC 点云淡入
- 网格球点云淡出（可选）

### Shot 2: FCC 网格球旋转
- `RotationPivot` 绕 Z 轴旋转 180°
- 坐标轴跟随旋转

### Shot 3: 与实验晶粒重叠
- 晶粒点云淡入
- FCC 球体平移到与晶粒重叠的位置

### Shot 4: 最终重叠结果
- 仅显示晶粒内部的 FCC 点（高亮显示）
- 晶粒点云透明度增加
- 坐标轴透明度略微降低

## 点云过滤（Shot 4）

在 Shot 4 中，脚本会过滤出位于晶粒内部的 FCC 点：

- 使用距离阈值判断点是否在晶粒内
- 默认阈值：`point_size * 2`
- 如果安装了 `scipy`，可以使用 `--use-scipy` 加速计算

## 渲染动画

### 在 Blender 中渲染

1. 打开生成的 `.blend` 文件
2. 切换到 **Rendered** 视图（按 `Z` → 选择 `Rendered`）
3. 检查动画时间轴（帧 1-260）
4. 设置输出路径：`Render Properties` → `Output` → 选择输出文件夹
5. 渲染动画：`Render` → `Render Animation`（或按 `Ctrl+F12`）

### 命令行渲染

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    workflow_animation.blend \
    --render-anim \
    --render-output //frames/frame_####.png
```

## 自定义和调整

### 调整动画速度

在 Blender 中：
1. 选择 `RotationPivot` 对象
2. 打开 `Graph Editor`
3. 调整旋转关键帧的插值曲线

### 调整材质颜色

在 Blender 中：
1. 打开 `Material Properties`
2. 选择要修改的材质（如 `MAT_GridDefault`）
3. 调整 `Base Color` 和 `Alpha` 值

### 调整相机位置

在 Blender 中：
1. 选择 `Camera` 对象
2. 调整位置和旋转
3. 使用 `Numpad 0` 切换到相机视图预览

## 性能优化

### 减少点云数量

如果点云过多导致性能问题：

1. 减小 `--grid-ball-radius`
2. 增大 `--grid-spacing`
3. 在脚本中修改 `generate_fcc_points` 函数，使用更少的点

### 使用 scipy 加速

安装 scipy：
```bash
pip install scipy
```

然后使用 `--use-scipy` 参数。

## 故障排除

### 点云不显示

- 确保在 **Rendered** 视图中查看（不是 Material Preview）
- 检查对象的 `hide_render` 属性
- 检查材质的 `Alpha` 值是否大于 0

### 坐标轴不显示

- 坐标轴使用网格对象（圆柱体和圆锥），应该可见
- 检查 `AxisHelper` 及其子对象的可见性
- 确保在 **Solid** 或 **Rendered** 视图中查看

### 动画不流畅

- 检查关键帧插值类型（应该是 `Linear` 或 `Bezier`）
- 减少点云数量
- 使用 Eevee 渲染器（比 Cycles 快）

## 输出文件

脚本会生成一个 `.blend` 文件，包含：
- 完整的场景设置
- 所有点云对象和材质
- 所有动画关键帧
- 相机和光照设置

你可以在 Blender 中打开这个文件，进行进一步的自定义和渲染。

## 示例工作流程

1. **生成点云数据**（如果还没有）：
   ```bash
   python export_grain_points.py --dream3d An0new6.dream3d --grain-id 111
   ```

2. **创建动画场景**：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender \
       --background \
       --python blender_animation_workflow.py \
       -- \
       --grain-ply grain_111_points.ply \
       --use-scipy
   ```

3. **在 Blender 中打开和调整**：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender workflow_animation.blend
   ```

4. **渲染动画**：
   - 在 Blender 中：`Render` → `Render Animation`
   - 或命令行：`blender --background workflow_animation.blend --render-anim`

## 注意事项

- 所有点云都使用球体实例化（不是立方体）
- 相机在整个动画过程中保持固定
- 所有旋转都应用于 `RotationPivot` 对象
- 坐标轴跟随 `RotationPivot` 旋转
- 背景为纯白色
- 无网格/地板显示
