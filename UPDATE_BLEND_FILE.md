# 如何更新 grain_111_render.blend 文件

## 更新内容

脚本 `blender_render.py` 已更新，包含以下改进：

1. **360 度旋转动画**：RotationPivot 现在旋转 360 度（而不是之前的 180 度）
2. **GrainVoxel 对象**：在 RotationPivot 下添加了 GrainVoxel 对象
   - 只保留最外层 surface（删除了所有内部相连的 surface）
   - 内部 voxel 合并成一个 volume（实心 mesh）

## 更新步骤

### 方法 1: 重新生成文件（推荐）

直接运行更新后的脚本重新生成 `.blend` 文件：

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_render.py \
    -- \
    --grain-ply grain_111_points.ply \
    --margin-ply grain_111_margin_points.ply \
    --voxel-size 1.0 \
    --output-blend grain_111_render.blend
```

这会：
- 创建新的 `grain_111_render.blend` 文件
- 应用 360 度旋转动画
- 自动创建 GrainVoxel 对象

### 方法 2: 在 Blender 中手动更新现有文件

如果你想保留 `grain_111_render_manual.blend` 中的手动调整，可以：

1. **打开 `grain_111_render_manual.blend`**
2. **更新 RotationPivot 动画**：
   - 选择 `RotationPivot` 对象
   - 打开 `Graph Editor` (按 `N` 或 `Window` → `Animation` → `Graph Editor`)
   - 找到 `rotation_euler` → `Z` 的动画曲线
   - 将最后一帧的值改为 `6.28318` (360 度 = 2π 弧度)
   - 确保插值类型为 `Linear`（线性）

3. **添加 GrainVoxel 对象**：
   - 选择 `GrainPoints` 对象
   - 复制它 (`Ctrl+D` 或 `Object` → `Duplicate`)
   - 重命名为 `GrainVoxel`
   - 应用 Geometry Nodes 修改器（`Modifier Properties` → `Apply`）
   - 进入 `Edit Mode` (`Tab`)
   - 选择所有 (`A`)
   - 删除内部面：
     - `Mesh` → `Select` → `Select Interior Faces`（如果可用）
     - 或使用 `Mesh` → `Select` → `Select Non-Manifold` → 选择边 → 转换为面选择 → 反选 → 删除
   - 退出 `Edit Mode` (`Tab`)
   - 将 `GrainVoxel` 的父对象设置为 `RotationPivot`

4. **保存为 `grain_111_render.blend`**

### 方法 3: 使用 Python 脚本更新现有文件

创建一个更新脚本：

```python
# update_blend.py
import bpy

# 打开现有文件
bpy.ops.wm.open_mainfile(filepath="grain_111_render.blend")

# 获取 RotationPivot
pivot = bpy.data.objects.get("RotationPivot")
if pivot:
    # 更新动画为 360 度
    if pivot.animation_data and pivot.animation_data.action:
        for fcurve in pivot.animation_data.action.fcurves:
            if fcurve.data_path == "rotation_euler" and fcurve.array_index == 2:
                # 找到最后一帧并更新为 360 度
                if len(fcurve.keyframe_points) > 0:
                    last_kf = fcurve.keyframe_points[-1]
                    last_kf.co[1] = 6.28318  # 360 degrees
                    last_kf.interpolation = 'LINEAR'

# 保存
bpy.ops.wm.save_as_mainfile(filepath="grain_111_render.blend")
```

运行：
```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python update_blend.py
```

## 验证更新

更新后，检查：

1. **RotationPivot 动画**：
   - 打开 `Graph Editor`
   - 查看 `rotation_euler[2]` 曲线
   - 第一帧应该是 `0`，最后一帧应该是 `6.28318` (360°)

2. **GrainVoxel 对象**：
   - 在 `Outliner` 中应该能看到 `GrainVoxel` 对象
   - 它应该是 `RotationPivot` 的子对象
   - 在 `Edit Mode` 中查看，应该只有外层表面，没有内部面

3. **对象层次结构**：
   ```
   RotationPivot
   ├── GrainPoints
   ├── MarginPoints (如果存在)
   └── GrainVoxel
   ```

## 注意事项

- **GrainVoxel 创建**：如果点云很大，创建 GrainVoxel 可能需要一些时间
- **内存使用**：GrainVoxel 会创建实际的 mesh，可能占用较多内存
- **渲染性能**：GrainVoxel 的渲染可能比点云慢，因为它是完整的 mesh

## 故障排除

### GrainVoxel 创建失败

如果 GrainVoxel 创建失败，可能的原因：
- 点云数据有问题
- Blender 版本不支持某些操作
- 内存不足

**解决方案**：
- 检查 `grain_111_points.ply` 文件是否正确
- 尝试减小 `--voxel-size` 参数
- 在 Blender 中手动创建 GrainVoxel（见方法 2）

### 动画不流畅

如果动画不流畅：
- 检查关键帧插值类型（应该是 `Linear`）
- 确保帧范围正确（`frame_start=1`, `frame_end=n_frames`）

### 看不到 GrainVoxel

如果看不到 GrainVoxel：
- 检查它是否被隐藏（`Outliner` 中眼睛图标）
- 检查材质透明度设置
- 确保在 `Rendered` 视图中查看
