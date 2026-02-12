# 提取和应用 Geometry Nodes 设置

## 步骤 1: 提取 Geometry Nodes 设置

### 方法 A: 在 Blender 中运行脚本（推荐）

1. **打开 `grain_111_render_manual.blend`** 在 Blender 中
2. **切换到 Scripting 工作区**（顶部标签）
3. **创建新脚本**（点击 `New`）
4. **复制并粘贴** `blender_read_geometry_nodes_internal.py` 的内容
5. **运行脚本**（点击 `Run Script` 或按 `Alt+P`）
6. **查看输出**：脚本会在控制台打印所有 Geometry Nodes 信息
7. **保存 JSON**：脚本会尝试保存到 `geometry_nodes_settings.json`

### 方法 B: 手动查看并记录

1. 选择 `GrainPoints` 对象
2. 打开 `Modifier Properties`（扳手图标）
3. 找到 `GeometryNodes` 修改器
4. 点击 `Geometry Nodes` 按钮进入节点编辑器
5. 记录所有节点的设置：
   - 节点类型
   - 输入参数值
   - 节点之间的连接

## 步骤 2: 应用设置到新文件

### 如果成功提取了 JSON 文件：

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    --background \
    --python blender_render.py \
    -- \
    --grain-ply grain_111_points.ply \
    --margin-ply grain_111_margin_points.ply \
    --voxel-size 1.0 \
    --geometry-nodes-config geometry_nodes_settings.json \
    --output-blend grain_111_render.blend
```

### 如果 JSON 文件包含多个配置：

脚本会自动使用第一个配置。如果你想指定使用哪个配置，可以：

1. 编辑 `geometry_nodes_settings.json`
2. 只保留你需要的配置（删除其他的）
3. 或者修改脚本以支持指定配置名称

## 步骤 3: 验证

更新后，检查：

1. **Geometry Nodes 设置**：
   - 选择 `GrainPoints` 对象
   - 查看 `Modifier Properties` → `GeometryNodes`
   - 点击进入节点编辑器
   - 验证所有节点和参数是否正确

2. **视觉效果**：
   - 在 `Rendered` 视图中查看
   - 确保 voxel 大小、间距等与 manual 版本一致

## 常见问题

### Q: 脚本无法读取某些节点类型

**A**: 某些自定义节点或 Blender 版本特定的节点可能无法完全读取。在这种情况下：
- 手动记录这些节点的设置
- 在脚本中手动添加这些节点

### Q: JSON 文件格式不正确

**A**: 检查 JSON 文件格式：
```bash
python -m json.tool geometry_nodes_settings.json
```

如果格式错误，修复后再运行。

### Q: 某些参数值无法应用

**A**: 可能的原因：
- 参数类型不匹配
- Blender 版本差异
- 节点接口变化

**解决方案**：
- 检查控制台的警告信息
- 手动调整无法自动应用的参数

## 手动调整（如果需要）

如果自动应用失败，可以：

1. 运行脚本生成基础结构
2. 在 Blender 中手动调整 Geometry Nodes
3. 保存文件

脚本已经创建了正确的节点结构，只需要调整参数值即可。
