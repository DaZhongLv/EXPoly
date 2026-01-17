# EXPoly 重构实施清单

## ✅ 完成状态跟踪

### 阶段 1: 仓库结构改进
- [ ] 1.1 清理未使用依赖 (`pyproject.toml`)
  - [ ] 移除 `scikit-learn`
  - [ ] 移除 `jinja2`
  - [ ] 验证 `plotly` 使用情况
- [ ] 1.2 创建 `docs/` 目录
  - [ ] `docs/use_cases.md`
  - [ ] `docs/architecture.md` (可选)
- [ ] 1.3 创建 `examples/` 目录
  - [ ] `examples/README.md`
  - [ ] `examples/minimal_example.py`
  - [ ] `examples/toy_data_generator.py`
- [ ] 1.4 添加元数据文件
  - [ ] `CHANGELOG.md`
  - [ ] `CITATION.cff`
  - [ ] `CONTRIBUTING.md`
- [ ] 1.5 检查 `LICENSE`

**验证命令**: `pip install -e . && expoly --help`

---

### 阶段 2: README.md 改进
- [ ] 2.1 添加价值主张（一行）
- [ ] 2.2 添加 30 秒快速开始
- [ ] 2.3 添加流程图（ASCII/Mermaid）
- [ ] 2.4 添加 "Failure modes & assumptions" 部分
- [ ] 2.5 改进 "Outputs" 部分

**验证命令**: 检查 README.md 可读性和完整性

---

### 阶段 3: 测试 + CI
- [ ] 3.1 创建测试目录
  - [ ] `tests/__init__.py`
  - [ ] `tests/conftest.py`
  - [ ] `tests/test_frames.py`
  - [ ] `tests/test_carve.py`
  - [ ] `tests/test_polish.py`
  - [ ] `tests/test_cli.py`
  - [ ] `tests/fixtures/`
- [ ] 3.2 实现核心测试
  - [ ] Frame 测试（HDF5 解析、grain ID 查询）
  - [ ] Carve 测试（确定性映射、晶格生成）
  - [ ] Polish 测试（LAMMPS 生成、文件存在性）
  - [ ] CLI 测试（参数解析、错误处理）
- [ ] 3.3 更新 `pyproject.toml` 测试依赖
- [ ] 3.4 创建 `.github/workflows/tests.yml`
- [ ] 3.5 添加代码格式化配置（ruff/black）

**验证命令**: `pytest tests/ -v`

---

### 阶段 4: CLI 改进
- [ ] 4.1 改进帮助信息（使用 ArgumentGroup）
- [ ] 4.2 实现 `expoly doctor` 子命令
  - [ ] 检查文件存在性
  - [ ] 检查 HDF5 数据集
  - [ ] 检查值范围
  - [ ] 检查 OVITO
  - [ ] 输出修复建议
- [ ] 4.3 改进错误消息（具体化）
- [ ] 4.4 实现 `pipeline.run()` 函数

**验证命令**: 
- `expoly run --help` (检查分组)
- `expoly doctor --dream3d <file>` (检查诊断)
- `python -c "from expoly import run; ..."` (检查 API)

---

### 阶段 5: 基准测试
- [ ] 5.1 创建 `benchmarks/` 目录
  - [ ] `benchmarks/README.md`
  - [ ] `benchmarks/benchmark.py`
  - [ ] `benchmarks/generate_toy_data.py`
- [ ] 5.2 实现基准测试脚本
- [ ] 5.3 在文档中记录基准结果

**验证命令**: `python benchmarks/benchmark.py`

---

## 快速验证脚本

创建 `scripts/validate.sh`:

```bash
#!/bin/bash
set -e

echo "=== 1. 安装检查 ==="
pip install -e . > /dev/null
expoly --help > /dev/null && echo "✓ CLI 可用"

echo "=== 2. 测试 ==="
pytest tests/ -v --tb=short

echo "=== 3. 示例 ==="
python examples/minimal_example.py && echo "✓ 示例运行成功"

echo "=== 4. Doctor ==="
# 需要测试文件
# expoly doctor --dream3d <test_file> && echo "✓ Doctor 可用"

echo "=== 5. 代码质量 ==="
ruff check . --quiet && echo "✓ 代码格式检查通过"

echo "=== 所有检查通过 ==="
```

---

## 提交策略

每个阶段完成后提交:

```bash
git add .
git commit -m "feat: [阶段 X] 描述

- 具体变更 1
- 具体变更 2
"
```

示例:
```bash
git commit -m "feat: [阶段 1] 仓库结构改进

- 移除未使用的 scikit-learn 和 jinja2 依赖
- 添加 docs/use_cases.md
- 添加 examples/ 目录和最小示例
- 添加 CHANGELOG.md, CITATION.cff, CONTRIBUTING.md
"
```

---

## 优先级调整

如果时间有限，按以下顺序实施:

1. **必须**: 阶段 1.1, 1.4, 阶段 2, 阶段 3.2 (最小测试), 阶段 4.2 (doctor)
2. **重要**: 阶段 1.3 (examples), 阶段 3.4 (CI), 阶段 4.1 (帮助改进)
3. **可选**: 阶段 1.2 (docs), 阶段 3.5 (格式化), 阶段 5 (基准测试)
