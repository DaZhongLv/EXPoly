# Paraview 和 pvpython 安装指南

`pvpython` 是 Paraview 自带的 Python 解释器，用于在非 GUI 模式下运行 Paraview Python 脚本。

## macOS 安装方法

### 方法 1: 使用 Homebrew（推荐）

最简单的方法是使用 Homebrew：

```bash
# 安装 Paraview（包含 pvpython）
brew install paraview

# 验证安装
pvpython --version
```

### 方法 2: 下载官方二进制版本

1. 访问 Paraview 官网：https://www.paraview.org/download/
2. 选择 macOS 版本下载（通常是一个 `.dmg` 文件）
3. 打开 `.dmg` 文件，将 `ParaView.app` 拖到 `Applications` 文件夹
4. `pvpython` 位于应用包内，路径通常是：
   ```
   /Applications/ParaView-<version>.app/Contents/bin/pvpython
   ```

5. 为了方便使用，可以创建符号链接或添加到 PATH：

```bash
# 找到 pvpython 的完整路径（替换版本号）
PARAVIEW_VERSION="5.12.0"  # 替换为你的版本号
sudo ln -s "/Applications/ParaView-${PARAVIEW_VERSION}.app/Contents/bin/pvpython" /usr/local/bin/pvpython

# 或者添加到 PATH（在 ~/.zshrc 或 ~/.bash_profile 中）
export PATH="/Applications/ParaView-${PARAVIEW_VERSION}.app/Contents/bin:$PATH"
```

### 方法 3: 使用 Conda（如果你使用 Conda）

```bash
conda install -c conda-forge paraview
```

## Linux 安装方法

### Ubuntu/Debian

```bash
# 使用 apt 安装
sudo apt update
sudo apt install paraview

# 或者下载官方二进制版本
# 从 https://www.paraview.org/download/ 下载 Linux 版本
# 解压后，pvpython 在 <解压目录>/bin/pvpython
```

### 安装依赖（如果需要）

```bash
sudo apt install '^libxcb.*' libx11-xcb libglu1-mesa libxrender libxi libxkbcommon libxkbcommon-x11
```

## Windows 安装方法

1. 访问 https://www.paraview.org/download/
2. 下载 Windows 安装程序（`.exe` 文件）
3. 运行安装程序，按照提示安装
4. `pvpython.exe` 通常位于：
   ```
   C:\Program Files\ParaView <version>\bin\pvpython.exe
   ```

5. 为了方便使用，可以将该目录添加到系统 PATH 环境变量中

## 验证安装

安装完成后，在终端中运行：

```bash
pvpython --version
```

如果显示版本号（例如 `ParaView 5.12.0`），说明安装成功。

## 测试 pvpython

可以运行一个简单的测试脚本：

```bash
pvpython -c "from paraview.simple import *; print('Paraview Python API works!')"
```

如果输出 "Paraview Python API works!"，说明一切正常。

## 使用 pvpython 运行脚本

安装完成后，你可以这样运行可视化脚本：

```bash
pvpython visualize_grain_mesh_paraview.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    --h5-grain-dset FeatureIds \
    --h5-euler-dset EulerAngles \
    --h5-numneighbors-dset NumNeighbors \
    --h5-neighborlist-dset NeighborList2 \
    --h5-dimensions-dset DIMENSIONS
```

## 常见问题

### 问题 1: 找不到 pvpython 命令

**解决方案：**
- 检查 Paraview 是否已正确安装
- 找到 `pvpython` 的完整路径，使用完整路径运行，或添加到 PATH

### 问题 2: 导入 paraview.simple 失败

**解决方案：**
- 确保使用 `pvpython` 而不是普通的 `python`
- `pvpython` 已经包含了 Paraview 的 Python 模块

### 问题 3: macOS 上权限问题

**解决方案：**
```bash
# 如果遇到权限问题，可能需要允许应用运行
# 在系统设置 > 安全性与隐私 > 允许从以下位置下载的应用
```

### 问题 4: 无头模式（服务器环境）

如果你在服务器上运行（没有图形界面），可以使用：

```bash
pvpython --mesa visualize_grain_mesh_paraview.py ...
```

或者设置环境变量：

```bash
export MESA_GL_VERSION_OVERRIDE=3.3
pvpython visualize_grain_mesh_paraview.py ...
```

## 备选方案

如果安装 Paraview 有困难，可以使用 matplotlib 版本的脚本：

```bash
python visualize_grain_mesh_matplotlib.py \
    --dream3d An0new6.dream3d \
    --grain-id 111 \
    ...
```

这个版本不需要 Paraview，只需要标准的 Python 包（matplotlib, scipy, scikit-image）。
