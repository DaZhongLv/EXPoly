# GitHub 上传完整指南

## 📋 上传前检查清单

### ✅ 确认所有文件已准备好

1. **核心文件**:
   - ✅ `pyproject.toml` (版本 1.0.0)
   - ✅ `LICENSE` (MIT)
   - ✅ `.gitignore` (已排除 *.dream3d)
   - ✅ `README.md` (包含 CMU 下载链接)

2. **源代码**: `src/expoly/` 所有文件
3. **测试**: `tests/` 所有文件
4. **文档**: `docs/`, `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`
5. **示例和基准**: `examples/`, `benchmarks/`
6. **CI**: `.github/workflows/tests.yml`

### ⚠️ 确认大文件被排除

- ✅ `An0new6.dream3d` (554 MB) 已被 `.gitignore` 排除
- ✅ 运行 `git check-ignore -v An0new6.dream3d` 应该显示被忽略

---

## 🚀 上传步骤

### 步骤 1: 初始化 Git 仓库（如果还没有）

```bash
# 检查是否已有 Git 仓库
if [ ! -d .git ]; then
    git init
    git branch -M main
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi
```

### 步骤 2: 检查 .gitignore 是否生效

```bash
# 检查大文件是否被忽略
git check-ignore -v An0new6.dream3d
# 应该输出: .gitignore:20:*.dream3d	An0new6.dream3d

# 查看会被添加的文件（预览）
git status
# 应该看不到 An0new6.dream3d
```

### 步骤 3: 添加所有文件

```bash
# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 再次检查将要提交的文件
git status
# 确认：
# - 所有代码文件都在
# - 所有文档都在
# - An0new6.dream3d **不在**（这是正确的）
```

### 步骤 4: 创建初始提交

```bash
git commit -m "feat: v1.0.0 - Professional refactoring release

Complete repository structure with:
- Comprehensive documentation and examples
- Full test suite (~25 tests) with pytest
- GitHub Actions CI workflow (Python 3.10/3.11)
- Improved CLI with grouped help and doctor command
- Programmatic API (pipeline.run)
- Benchmarking infrastructure
- Enhanced error messages and diagnostics
- Sample data download links (CMU Grain Boundary Archive)

See CHANGELOG.md for details."
```

### 步骤 5: 在 GitHub 上创建仓库

1. **登录 GitHub**: https://github.com
2. **点击右上角 "+"** → "New repository"
3. **填写仓库信息**:
   - Repository name: `EXPoly` (或你喜欢的名字)
   - Description: `Voxel-to-atomistic conversion and LAMMPS pipeline tools`
   - Visibility: Public 或 Private（根据你的需要）
   - **重要**: **不要**勾选以下选项：
     - ❌ Add a README file（我们已经有了）
     - ❌ Add .gitignore（我们已经有了）
     - ❌ Choose a license（我们已经有了）
4. **点击 "Create repository"**

### 步骤 6: 连接本地仓库到 GitHub

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git

# 或者使用 SSH（如果你配置了 SSH keys）
# git remote add origin git@github.com:YOUR_USERNAME/EXPoly.git

# 验证远程仓库
git remote -v
# 应该显示:
# origin  https://github.com/YOUR_USERNAME/EXPoly.git (fetch)
# origin  https://github.com/YOUR_USERNAME/EXPoly.git (push)
```

### 步骤 7: 推送代码到 GitHub

```bash
# 推送 main 分支
git push -u origin main

# 如果遇到错误（比如分支名是 master），使用：
# git push -u origin main:main
# 或者先重命名分支：
# git branch -M main
# git push -u origin main
```

### 步骤 8: 验证上传

1. **访问你的 GitHub 仓库**: `https://github.com/YOUR_USERNAME/EXPoly`
2. **检查文件**:
   - ✅ 所有代码文件都在
   - ✅ 所有文档都在
   - ✅ `.dream3d` 文件**不在**（这是正确的）
   - ✅ `.gitignore` 文件存在
3. **检查 GitHub Actions**:
   - 如果有 push，CI 应该会自动运行
   - 查看 Actions 标签页确认测试是否通过

---

## 🔧 常见问题解决

### 问题 1: "fatal: remote origin already exists"

```bash
# 删除现有远程仓库
git remote remove origin

# 重新添加
git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git
```

### 问题 2: "error: failed to push some refs"

```bash
# 如果远程仓库有内容（比如 README），先拉取
git pull origin main --allow-unrelated-histories

# 或者强制推送（谨慎使用）
# git push -u origin main --force
```

### 问题 3: 分支名不匹配（master vs main）

```bash
# 重命名本地分支
git branch -M main

# 推送
git push -u origin main
```

### 问题 4: 需要输入 GitHub 凭证

**选项 A: 使用 Personal Access Token (推荐)**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. 选择权限: `repo` (完整仓库访问)
4. 复制 token
5. 推送时使用 token 作为密码

**选项 B: 使用 SSH**
```bash
# 生成 SSH key（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到 GitHub: Settings → SSH and GPG keys → New SSH key
# 然后使用 SSH URL:
git remote set-url origin git@github.com:YOUR_USERNAME/EXPoly.git
```

---

## 📊 上传后验证清单

- [ ] 所有源代码文件都在仓库中
- [ ] 所有文档文件都在仓库中
- [ ] `.gitignore` 文件存在
- [ ] `An0new6.dream3d` **不在**仓库中（正确）
- [ ] README.md 显示正确（包含 CMU 下载链接）
- [ ] GitHub Actions CI 工作流运行成功
- [ ] 仓库描述和标签设置正确

---

## 🎯 快速命令（一键执行）

如果你已经确认所有文件都准备好了，可以运行：

```bash
# 1. 初始化（如果需要）
[ ! -d .git ] && git init && git branch -M main

# 2. 检查大文件
git check-ignore -v An0new6.dream3d || echo "⚠️ 大文件未被忽略，请检查 .gitignore"

# 3. 添加文件
git add .

# 4. 检查将要提交的文件
echo "=== 将要提交的文件 ==="
git status --short

# 5. 创建提交
git commit -m "feat: v1.0.0 - Professional refactoring release

Complete repository structure with comprehensive documentation,
test suite, CI workflow, improved CLI, and benchmarking infrastructure."

# 6. 添加远程（替换 YOUR_USERNAME）
# git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git

# 7. 推送
# git push -u origin main
```

**注意**: 步骤 6 和 7 需要你先在 GitHub 创建仓库，然后替换 `YOUR_USERNAME`。

---

## 📝 后续维护

### 更新代码后推送

```bash
git add .
git commit -m "描述你的更改"
git push
```

### 创建 Release

1. GitHub 仓库页面 → Releases → "Create a new release"
2. Tag: `v1.0.0`
3. Title: `v1.0.0 - Professional Refactoring Release`
4. Description: 从 `CHANGELOG.md` 复制内容
5. 发布

### 添加 Topics/Tags

在仓库页面点击 ⚙️ → Topics，添加：
- `materials-science`
- `molecular-dynamics`
- `lammps`
- `dream3d`
- `microstructure`
- `atomistic-simulation`

---

## ✅ 完成！

上传完成后，你的仓库应该：
- ✅ 包含所有代码和文档
- ✅ 有清晰的 README 和示例
- ✅ 有 CI 工作流自动运行测试
- ✅ 排除大文件（通过 .gitignore）
- ✅ 提供 sample 数据下载链接

用户现在可以：
1. 克隆仓库
2. 从 CMU 下载 sample 数据
3. 或使用 toy data generator
4. 运行示例和测试
