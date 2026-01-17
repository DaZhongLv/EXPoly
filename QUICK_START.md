# EXPoly 快速开始指南

## ✅ 文件完整性确认

所有文件已保存完成！以下是验证和开始使用的步骤。

## 📋 文件检查

### 核心文件（已确认）

✅ **项目配置**
- `pyproject.toml` (版本 1.0.0)
- `LICENSE` (MIT)
- `.gitignore` (已配置排除 *.dream3d)

✅ **源代码** (src/expoly/)
- 所有 8 个 Python 模块文件

✅ **测试** (tests/)
- 4 个测试文件，~25 个测试

✅ **文档**
- README.md (完整，328 行)
- docs/use_cases.md
- docs/benchmarks.md

✅ **示例和基准**
- examples/ (2 个示例文件)
- benchmarks/ (2 个基准脚本)

✅ **CI/CD**
- .github/workflows/tests.yml

## 🚀 开始使用

### 1. 本地安装和验证

```bash
# 进入项目目录
cd /Users/lvmeizhong/Desktop/expoly-with-legacy/EXPoly

# 安装（开发模式）
pip install -e ".[dev]"
pip install ovito

# 验证安装
expoly --help

# 运行测试
pytest tests/ -v

# 测试 doctor 命令
expoly doctor --dream3d An0new6.dream3d --hx 0:50 --hy 0:50 --hz 0:50
```

### 2. 运行示例

**选项 A: 下载真实 sample 数据**
```bash
# 从 CMU Grain Boundary Data Archive 下载
# 访问: http://mimp.materials.cmu.edu/~gr20/Grain_Boundary_Data_Archive/Ni_velocity/Ni_velocity.html
# 下载 "Microstructure Data" archive (367 MB, 包含 6 个 Dream3D 文件)
# 解压后使用任意 Dream3D 文件
```

**选项 B: 使用本地 sample 文件（如果存在）**
```bash
cd examples
python minimal_example.py
# 会自动检测并使用 An0new6.dream3d（如果存在）
```

**选项 C: 生成测试数据**
```bash
# 生成小测试文件
python examples/toy_data_generator.py

# 运行示例（会使用生成的 toy_data.dream3d）
python examples/minimal_example.py
```

### 3. 使用自己的数据

```bash
expoly run \
  --dream3d /path/to/your_data.dream3d \
  --hx 0:100 --hy 0:100 --hz 0:100 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524
```

## 📤 GitHub 上传

### 关于 Sample 文件（An0new6.dream3d）

**重要**: 该文件约 **554 MB**，太大无法直接提交到 GitHub。

**解决方案**:
1. ✅ **已配置**: `.gitignore` 已排除 `*.dream3d` 文件
2. ✅ **自动处理**: Git 会自动忽略该文件，不会上传
3. ✅ **替代方案**: 用户可以使用 `toy_data_generator.py` 生成小测试文件

### 上传步骤

```bash
# 1. 检查 Git 状态（确认 .dream3d 被忽略）
git status
# 应该看不到 An0new6.dream3d

# 2. 初始化仓库（如果还没有）
git init
git branch -M main

# 3. 添加文件
git add .
git status  # 再次确认没有大文件

# 4. 创建初始提交
git commit -m "feat: v1.0.0 - Professional refactoring release

Complete repository structure with:
- Comprehensive documentation and examples
- Full test suite with pytest
- GitHub Actions CI workflow
- Improved CLI with doctor command
- Programmatic API (pipeline.run)
- Benchmarking infrastructure"

# 5. 在 GitHub 创建仓库后，连接并推送
git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git
git push -u origin main
```

### 验证上传

上传后，确认：
- ✅ 所有代码文件都在
- ✅ 所有文档都在
- ✅ `.dream3d` 文件**不在**仓库中（这是正确的）
- ✅ `.gitignore` 文件存在

## 📝 文件清单

### 应该上传的文件（< 5 MB 总计）

**源代码**:
- `src/expoly/*.py` (8 个文件)

**测试**:
- `tests/*.py` (5 个文件)

**文档**:
- `README.md`
- `CHANGELOG.md`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `LICENSE`
- `docs/*.md` (2 个文件)

**配置**:
- `pyproject.toml`
- `.gitignore`
- `.github/workflows/tests.yml`

**示例和基准**:
- `examples/*.py` (2 个文件)
- `benchmarks/*.py` (2 个文件)

### 不应该上传的文件（已排除）

- ❌ `An0new6.dream3d` (554 MB - 太大)
- ❌ `__pycache__/` (Python 缓存)
- ❌ `*.egg-info/` (构建文件)
- ❌ `.venv/` (虚拟环境)
- ❌ `runs/` (输出目录)

## 🎯 下一步

1. ✅ **验证本地**: 运行测试和示例
2. ⏭️ **初始化 Git**: `git init` (如果还没有)
3. ⏭️ **创建 GitHub 仓库**: 在 GitHub 上创建新仓库
4. ⏭️ **推送代码**: `git push`
5. ⏭️ **处理 sample 文件**: 
   - 选项 A: 保持排除（推荐，用户自己提供数据）
   - 选项 B: 使用 Git LFS（如果必须包含）
   - 选项 C: 上传到 GitHub Releases

## 📚 相关文档

- `GITHUB_SETUP.md` - 详细的 GitHub 设置指南
- `FILE_CHECKLIST.md` - 完整的文件检查清单
- `README.md` - 主文档（包含所有使用说明）
