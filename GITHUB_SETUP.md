# GitHub 设置指南

## 📋 文件检查清单

### ✅ 核心文件（已确认存在）

**项目配置**:
- ✅ `pyproject.toml` - 项目配置（版本 1.0.0）
- ✅ `LICENSE` - MIT 许可证
- ✅ `.gitignore` - Git 忽略规则（已排除 *.dream3d）

**文档**:
- ✅ `README.md` - 主文档（完整）
- ✅ `CHANGELOG.md` - 版本变更记录
- ✅ `CITATION.cff` - 引用信息
- ✅ `CONTRIBUTING.md` - 贡献指南
- ✅ `docs/use_cases.md` - 用例说明
- ✅ `docs/benchmarks.md` - 基准测试文档

**源代码**:
- ✅ `src/expoly/` - 所有 Python 模块
- ✅ `src/expoly/__init__.py` - 包初始化
- ✅ `src/expoly/cli.py` - CLI（包含 doctor 命令）
- ✅ `src/expoly/pipeline.py` - 程序化 API

**测试**:
- ✅ `tests/` - 测试套件（~25 个测试）
- ✅ `.github/workflows/tests.yml` - CI 工作流

**示例和基准**:
- ✅ `examples/` - 示例代码
- ✅ `benchmarks/` - 基准测试脚本

### ⚠️ 关于 Sample 文件（An0new6.dream3d）

**问题**: Dream3D HDF5 文件通常很大（可能几十 MB 到几 GB），不适合直接提交到 GitHub。

**解决方案**:

1. **使用 .gitignore（已配置）**
   - `.gitignore` 已包含 `*.dream3d`，文件不会被提交
   - 这是推荐的做法

2. **替代方案**:
   - **选项 A**: 使用 Git LFS（Large File Storage）
     ```bash
     # 安装 Git LFS
     git lfs install
     git lfs track "*.dream3d"
     git add .gitattributes
     git add An0new6.dream3d
     ```
   - **选项 B**: 使用外部存储（推荐）
     - 将 sample 文件上传到云存储（Google Drive, Dropbox, 等）
     - 在 README 中提供下载链接
     - 或使用 GitHub Releases 上传
   - **选项 C**: 只提交小文件
     - 使用 `examples/toy_data_generator.py` 生成小测试文件
     - 这些文件足够小，可以提交

3. **当前建议**:
   - ✅ 保持 `.gitignore` 排除 `*.dream3d`
   - ✅ 在 README 中说明如何获取示例数据
   - ✅ 提供 `toy_data_generator.py` 作为替代

---

## 🚀 GitHub 上传步骤

### 1. 初始化 Git 仓库（如果还没有）

```bash
cd /Users/lvmeizhong/Desktop/expoly-with-legacy/EXPoly

# 检查是否已有 .git
if [ ! -d .git ]; then
    git init
    git branch -M main
fi
```

### 2. 检查 .gitignore

确认 `.gitignore` 包含：
```
*.dream3d
*.h5
*.hdf5
runs/
__pycache__/
*.egg-info/
.venv/
```

### 3. 添加文件到 Git

```bash
# 查看会被添加的文件（预览）
git status

# 添加所有文件（.gitignore 会自动排除）
git add .

# 检查将要提交的文件
git status
```

### 4. 创建初始提交

```bash
git commit -m "feat: v1.0.0 - Professional refactoring release

- Complete repository structure with docs and examples
- Comprehensive test suite with pytest
- GitHub Actions CI workflow
- Improved CLI with doctor command
- Programmatic API (pipeline.run)
- Benchmarking infrastructure
- Enhanced error messages and diagnostics"
```

### 5. 在 GitHub 上创建仓库

1. 登录 GitHub
2. 点击 "New repository"
3. 仓库名: `EXPoly`（或你喜欢的名字）
4. 描述: "Voxel-to-atomistic conversion and LAMMPS pipeline tools"
5. **不要**初始化 README、.gitignore 或 license（我们已经有了）
6. 点击 "Create repository"

### 6. 连接并推送

```bash
# 添加远程仓库（替换 YOUR_USERNAME）
git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git

# 或者使用 SSH
# git remote add origin git@github.com:YOUR_USERNAME/EXPoly.git

# 推送代码
git push -u origin main
```

---

## 📝 关于 Sample 文件的处理建议

### 推荐方案：README 说明 + 生成器

在 README 中添加说明：

```markdown
## Sample Data

The repository includes a sample Dream3D file (`An0new6.dream3d`) for testing.
However, due to file size limitations, it is not included in the repository.

**Options to get started:**

1. **Use your own data**: Provide your Dream3D HDF5 file
2. **Generate toy data**: Use the included generator:
   ```bash
   python examples/toy_data_generator.py
   ```
3. **Download sample** (if available): Check [Releases](../../releases) for sample data

For large files, consider using Git LFS or external storage.
```

### 或者使用 Git LFS（如果文件 < 100MB）

```bash
# 安装 Git LFS
brew install git-lfs  # macOS
# 或: https://git-lfs.github.com/

# 初始化
git lfs install

# 跟踪 .dream3d 文件
git lfs track "*.dream3d"
git add .gitattributes

# 添加文件
git add An0new6.dream3d
git commit -m "Add sample data via Git LFS"
```

**注意**: GitHub 免费账户的 Git LFS 有配额限制（1 GB 存储，1 GB/月 带宽）。

---

## ✅ 上传前最终检查

运行以下命令检查：

```bash
# 1. 检查 .gitignore 是否生效
git status --ignored | grep dream3d
# 应该显示 An0new6.dream3d 被忽略

# 2. 检查将要提交的文件
git status
# 确认没有意外的大文件

# 3. 检查文件大小
find . -type f -size +10M ! -path "./.git/*" ! -path "./.venv/*"
# 应该没有大文件（除了可能被忽略的 .dream3d）

# 4. 验证关键文件存在
ls -la README.md pyproject.toml LICENSE .github/workflows/tests.yml
```

---

## 🎯 快速开始（上传后）

用户克隆仓库后：

```bash
# 1. 克隆
git clone https://github.com/YOUR_USERNAME/EXPoly.git
cd EXPoly

# 2. 安装
pip install -e ".[dev]"
pip install ovito

# 3. 生成测试数据（如果没有真实数据）
python examples/toy_data_generator.py

# 4. 运行示例
python examples/minimal_example.py

# 5. 或使用自己的数据
expoly run \
  --dream3d your_data.dream3d \
  --hx 0:50 --hy 0:50 --hz 0:50 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524
```

---

## 📊 文件大小参考

- **代码文件**: < 1 MB（所有 Python 文件）
- **文档**: < 1 MB（所有 Markdown 文件）
- **测试数据**: < 1 MB（toy data generator 生成的小文件）
- **Sample .dream3d**: 可能几十 MB 到几 GB（应排除或使用 LFS）

---

## 🔒 隐私和许可证

- ✅ `LICENSE` 已设置为 MIT
- ✅ `CITATION.cff` 包含引用信息
- ✅ 确保没有敏感信息（API keys, 个人数据等）

---

## 📌 下一步

1. ✅ 检查所有文件已保存
2. ✅ 确认 .gitignore 正确配置
3. ⏭️ 初始化 Git 仓库（如果还没有）
4. ⏭️ 创建 GitHub 仓库
5. ⏭️ 推送代码
6. ⏭️ 处理 sample 文件（可选：Git LFS 或外部存储）
