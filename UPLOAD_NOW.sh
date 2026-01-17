#!/bin/bash
# EXPoly GitHub 上传脚本
# 使用方法: bash UPLOAD_NOW.sh

set -e  # 遇到错误立即退出

echo "=========================================="
echo "EXPoly GitHub 上传脚本"
echo "=========================================="
echo ""

# 1. 检查 Git 状态
echo "📋 步骤 1: 检查 Git 状态..."
if [ ! -d .git ]; then
    echo "❌ 错误: 这不是一个 Git 仓库"
    echo "   请先运行: git init"
    exit 1
fi

# 2. 检查大文件是否被忽略
echo ""
echo "📋 步骤 2: 检查大文件是否被忽略..."
if git check-ignore -v An0new6.dream3d > /dev/null 2>&1; then
    echo "✅ An0new6.dream3d 已被 .gitignore 正确排除"
else
    echo "⚠️  警告: An0new6.dream3d 未被忽略，请检查 .gitignore"
    read -p "   是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 3. 显示将要添加的文件
echo ""
echo "📋 步骤 3: 显示将要添加的文件..."
echo "--- 修改的文件 ---"
git status --short | grep "^ M" || echo "无修改文件"
echo ""
echo "--- 新文件 ---"
git status --short | grep "^??" || echo "无新文件"
echo ""

# 4. 确认是否继续
read -p "是否继续添加所有文件并提交？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 5. 添加所有文件
echo ""
echo "📋 步骤 4: 添加所有文件..."
git add .
echo "✅ 文件已添加到暂存区"

# 6. 再次检查（确认大文件不在）
echo ""
echo "📋 步骤 5: 确认大文件不在暂存区..."
if git diff --cached --name-only | grep -q "\.dream3d$"; then
    echo "❌ 错误: 发现 .dream3d 文件在暂存区！"
    echo "   请检查 .gitignore 配置"
    exit 1
else
    echo "✅ 确认: 大文件不在暂存区"
fi

# 7. 创建提交
echo ""
echo "📋 步骤 6: 创建提交..."
COMMIT_MSG="feat: v1.0.0 - Professional refactoring release

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

git commit -m "$COMMIT_MSG"
echo "✅ 提交已创建"

# 8. 检查远程仓库
echo ""
echo "📋 步骤 7: 检查远程仓库..."
if git remote get-url origin > /dev/null 2>&1; then
    REMOTE_URL=$(git remote get-url origin)
    echo "✅ 远程仓库: $REMOTE_URL"
else
    echo "❌ 错误: 未配置远程仓库"
    echo "   请运行: git remote add origin <your-repo-url>"
    exit 1
fi

# 9. 确认推送
echo ""
read -p "是否推送到 GitHub？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消推送。你可以稍后运行: git push -u origin main"
    exit 0
fi

# 10. 推送
echo ""
echo "📋 步骤 8: 推送到 GitHub..."
BRANCH=$(git branch --show-current)
echo "当前分支: $BRANCH"

if git push -u origin "$BRANCH"; then
    echo ""
    echo "=========================================="
    echo "✅ 成功！代码已推送到 GitHub"
    echo "=========================================="
    echo ""
    echo "查看仓库: $REMOTE_URL"
    echo ""
else
    echo ""
    echo "❌ 推送失败。可能的原因:"
    echo "   1. 需要配置 SSH keys 或 Personal Access Token"
    echo "   2. 远程仓库有新的提交，需要先 pull"
    echo "   3. 网络问题"
    echo ""
    echo "可以尝试:"
    echo "   git pull origin $BRANCH --rebase"
    echo "   git push -u origin $BRANCH"
    exit 1
fi
