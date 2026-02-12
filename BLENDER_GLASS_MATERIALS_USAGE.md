# Blender 玻璃材质设置脚本使用指南

## 功能

`blender_setup_glass_materials.py` 为 GrainPoints 和 MarginPoints 创建玻璃材质，包含：
- **玻璃效果**：使用 Principled BSDF，Transmission=1.0, IOR=1.45
- **边线增强**：使用 Wireframe 节点生成 voxel 边框效果
- **颜色区分**：
  - Grain: 淡灰玻璃 + 深灰边线 (#3A3A3C)
  - Margin: 淡蓝玻璃 + 深蓝灰边线 (#3B6A8C)
- **透明度**：Alpha Hashed 模式，无阴影

## macOS 运行方法

### 方法 1: 命令行运行（推荐）

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
    -P blender_setup_glass_materials.py
```

### 方法 2: 在 Blender 中运行

1. 打开 Blender，加载你的 `.blend` 文件
2. 切换到 **Scripting** 工作区
3. File → Open → 选择 `blender_setup_glass_materials.py`
4. 点击 **Run Script** 按钮（或按 Alt+P）

## 脚本功能

### 创建的材质

1. **MAT_Grain**
   - 玻璃颜色：淡灰 (#8E8E93)
   - 边线颜色：深灰 (#3A3A3C)
   - 透明度：0.38

2. **MAT_Margin**
   - 玻璃颜色：淡蓝 (#A7C7E7)
   - 边线颜色：深蓝灰 (#3B6A8C)
   - 透明度：0.14

### 材质节点结构

```
Geometry → Wireframe → ColorRamp → Mix (edge color + base color) → Principled BSDF → Output
```

### 自动配置

- **Eevee 渲染器**：自动启用 Screen Space Reflections 和 Refraction
- **Cycles 渲染器**：可选 Volume Absorption（增加玻璃厚度感）
- **材质应用**：自动应用到 Geometry Nodes 的 Realize Instances 输出

## 可调参数

在脚本顶部可以调整以下参数：

```python
# 边线宽度
WIRE_SIZE = 0.03  # 0.02~0.05，值越小边线越细

# 边线强度
EDGE_STRENGTH = 1.0  # 0.0~1.0，控制边线颜色混合强度

# 玻璃粗糙度
GLASS_ROUGHNESS = 0.1  # 0.0~1.0，值越大越模糊

# 透明度
GRAIN_ALPHA = 0.38
MARGIN_ALPHA = 0.14

# 体积吸收（Cycles）
USE_VOLUME_ABSORPTION = True
ABSORPTION_DENSITY = 0.1  # 增加玻璃颜色深度
```

## 使用步骤

1. **确保对象存在**：
   - `GrainPoints` 对象
   - `MarginPoints` 对象
   - 两者都有 Geometry Nodes modifier

2. **运行脚本**：
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender -P blender_setup_glass_materials.py
   ```

3. **在 Blender 中查看**：
   - 切换到 **Material Preview** 或 **Rendered** 视图
   - 应该能看到玻璃效果和边线

## 渲染器设置

### Eevee

脚本会自动配置：
- Screen Space Reflections (SSR)
- Screen Space Refraction
- 材质中的 Screen Space Refraction 选项

### Cycles

可选功能：
- Volume Absorption（在脚本中启用）
- 调整采样数以获得更好质量

## 故障排除

### 材质没有应用

如果材质没有正确应用：
1. 检查对象名称是否为 `GrainPoints` 和 `MarginPoints`
2. 确保 Geometry Nodes modifier 存在
3. 手动在 Material Properties 中检查材质

### 边线不可见

1. 增加 `WIRE_SIZE` 值（例如 0.05）
2. 增加 `EDGE_STRENGTH` 值
3. 检查 ColorRamp 节点设置

### 玻璃效果不明显

1. 切换到 **Rendered** 视图（Eevee 或 Cycles）
2. 确保渲染器设置正确（Eevee 需要 SSR 和 Refraction）
3. 调整 `GLASS_ROUGHNESS`（值越小越清晰）

## 手动设置步骤（如果脚本失败）

如果脚本无法运行，可以手动设置：

1. **创建材质**：
   - Material Properties → New Material
   - 命名为 `MAT_Grain`

2. **设置 Principled BSDF**：
   - Base Color: #8E8E93
   - Transmission: 1.0
   - IOR: 1.45
   - Roughness: 0.1
   - Alpha: 0.38

3. **添加 Wireframe 节点**：
   - Add → Input → Wireframe
   - Size: 0.03

4. **添加 ColorRamp**：
   - Add → Converter → ColorRamp
   - 连接 Wireframe Fac → ColorRamp Fac

5. **混合边线颜色**：
   - Add → Color → Mix RGB
   - Color1: 玻璃颜色
   - Color2: 边线颜色 (#3A3A3C)
   - Fac: 连接 ColorRamp Color

6. **应用材质**：
   - 选择对象
   - Material Properties → Assign

## 示例命令

```bash
# 基本运行
/Applications/Blender.app/Contents/MacOS/Blender -P blender_setup_glass_materials.py

# 运行并打开特定文件
/Applications/Blender.app/Contents/MacOS/Blender \
    grain_111_render.blend \
    -P blender_setup_glass_materials.py
```
