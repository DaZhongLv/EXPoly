# Blender 工作流程指南

## 避免脚本与手动修改冲突

### 推荐工作流程

#### 方法 1: 使用不同的文件名（推荐）

1. **初始生成**（使用脚本）：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender \
       grain_111_render.blend \
       -P blender_setup_glass_materials.py
   ```
   这会生成 `grain_111_render.blend`

2. **手动修改后保存**：
   - 在 Blender 中打开 `grain_111_render.blend`
   - 修改材质、光照、相机等
   - **另存为**：`grain_111_render_manual.blend` 或 `grain_111_render_final.blend`
   - 这样原始文件保持不变，脚本可以随时重新运行

#### 方法 2: 重命名材质

如果需要在同一个文件中保留手动修改的版本：

1. **在 Blender 中**：
   - Material Properties → 选择 `MAT_Grain`
   - 点击名称旁边的 `F2` 或双击名称
   - 重命名为 `MAT_Grain_Manual`
   - 同样处理 `MAT_Margin` → `MAT_Margin_Manual`

2. **重新运行脚本**：
   - 脚本会创建新的 `MAT_Grain` 和 `MAT_Margin`
   - 你的手动版本 `MAT_Grain_Manual` 和 `MAT_Margin_Manual` 会保留

#### 方法 3: 使用 Blender 的版本控制

1. **保存增量版本**：
   - `grain_111_render_v1.blend` - 初始脚本生成
   - `grain_111_render_v2.blend` - 第一次手动修改
   - `grain_111_render_v3.blend` - 进一步修改
   - 等等...

2. **脚本始终生成到新文件**：
   ```bash
   # 生成到新文件
   /Applications/Blender.app/Contents/MacOS/Blender \
       -P blender_setup_glass_materials.py
   # 然后另存为不同的文件名
   ```

## 最佳实践

### 1. 文件命名约定

```
grain_111_render.blend          # 脚本生成的原始文件
grain_111_render_manual.blend    # 手动修改后的文件
grain_111_render_final.blend     # 最终版本
```

### 2. 材质命名约定

如果要在同一文件中保留多个版本：

```
MAT_Grain          # 脚本生成的（可被覆盖）
MAT_Grain_Manual   # 手动修改的（保留）
MAT_Grain_V2       # 另一个版本
```

### 3. 工作流程建议

**推荐流程**：

1. **生成初始文件**：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender \
       -P blender_render.py -- --grain-ply ... --margin-ply ...
   ```
   生成 `grain_111_render.blend`

2. **设置材质**：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender \
       grain_111_render.blend \
       -P blender_setup_glass_materials.py
   ```

3. **手动调整**：
   - 在 Blender 中打开文件
   - 调整材质参数、光照、相机等
   - **另存为** `grain_111_render_manual.blend`

4. **后续修改**：
   - 如果需要重新生成，运行脚本（会覆盖原始文件）
   - 手动修改的文件不受影响

### 4. 脚本行为说明

当前脚本的行为：
- ✅ **会覆盖**已存在的 `MAT_Grain` 和 `MAT_Margin`
- ✅ **会警告**如果材质已存在
- ✅ **不会影响**其他材质（如 `MAT_Grain_Manual`）

### 5. 安全修改脚本（可选）

如果你想修改脚本，使其不覆盖已存在的材质，可以：

1. **修改脚本**，添加检查：
   ```python
   if "MAT_Grain" in bpy.data.materials:
       print("MAT_Grain already exists, skipping...")
   else:
       grain_mat = create_glass_material_with_edges(...)
   ```

2. **或使用不同的材质名**：
   ```python
   # 在脚本中修改
   grain_mat = create_glass_material_with_edges(
       name="MAT_Grain_Auto",  # 使用不同的名称
       ...
   )
   ```

## 常见场景

### 场景 1: 我想保留手动修改

**解决方案**：另存为新文件
```
grain_111_render.blend → grain_111_render_manual.blend
```

### 场景 2: 我想在同一文件中保留多个版本

**解决方案**：重命名现有材质
```
MAT_Grain → MAT_Grain_Manual
然后运行脚本创建新的 MAT_Grain
```

### 场景 3: 我想定期重新生成但保留手动版本

**解决方案**：使用版本号
```
grain_111_render_v1.blend  # 脚本生成
grain_111_render_v1_manual.blend  # 手动修改
grain_111_render_v2.blend  # 重新生成
```

## 总结

**最简单的做法**：
1. 脚本生成的文件：`grain_111_render.blend`
2. 手动修改后：另存为 `grain_111_render_manual.blend`
3. 这样两个文件互不干扰，可以随时重新运行脚本
