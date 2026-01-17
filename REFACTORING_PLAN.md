# EXPoly 专业化重构计划

## 目标
将 EXPoly 重构为研究级 Python 包，提升可维护性、可测试性和可用性，同时保持科学算法不变。

## 原则
- ✅ 保持向后兼容（CLI 参数不变）
- ✅ 增量式改进（小步提交）
- ✅ 优先工程信号（测试、文档、CI）
- ✅ 使用标准工具（pytest, GitHub Actions, ruff）
- ❌ 不重写科学算法
- ❌ 不引入重型依赖

---

## 阶段 1: 仓库结构改进

### 1.1 清理未使用依赖
**文件**: `pyproject.toml`
- 移除 `scikit-learn`（代码中未使用）
- 移除 `jinja2`（代码中未使用）
- 验证 `plotly` 是否必需（`general_func.py` 中可选）

**验证**: `pip install -e .` 后运行 `expoly --help`

---

### 1.2 创建文档目录结构
**新建文件**:
- `docs/use_cases.md` - 用例与假设/限制
- `docs/architecture.md` - 架构概览（可选）

**验证**: 文件存在

---

### 1.3 创建示例目录
**新建目录**: `examples/`
**新建文件**:
- `examples/README.md` - 示例说明
- `examples/minimal_example.py` - 最小可运行示例（使用合成数据）
- `examples/toy_data_generator.py` - 生成玩具数据（小 Dream3D HDF5 模拟）

**验证**: `python examples/minimal_example.py` 可运行

---

### 1.4 添加项目元数据文件
**新建文件**:
- `CHANGELOG.md` - 版本变更记录（从 v0.2.0 开始）
- `CITATION.cff` - 引用信息
- `CONTRIBUTING.md` - 贡献指南（最小版本）

**验证**: 文件存在且格式正确

---

### 1.5 检查 LICENSE
**文件**: `LICENSE`
- 确认 MIT 许可证存在且完整

**验证**: 文件存在

---

## 阶段 2: README.md 改进

### 2.1 添加价值主张
**文件**: `README.md`
- 在开头添加一行价值主张
- 格式: "EXPoly converts experimental microstructure voxel data (Dream3D HDF5) into MD-ready atomistic structures (LAMMPS data files)."

---

### 2.2 添加 30 秒快速开始
**文件**: `README.md`
- 在现有安装说明前添加 "Quickstart" 部分
- 包含: `pip install -e .` + 一个最小 CLI 命令示例

---

### 2.3 添加流程图
**文件**: `README.md`
- 使用 ASCII 或 Mermaid 绘制输入→输出流程
- 阶段: Dream3D HDF5 → Carve (lattice) → Polish (OVITO) → LAMMPS data

---

### 2.4 添加失败模式与假设
**文件**: `README.md`
- 新增 "Failure modes & assumptions" 部分
- 包含:
  - Schema 不匹配（HDF5 数据集路径）
  - FCC/BCC/DIA 差异
  - 大文件内存/时间限制
  - OVITO 依赖要求

---

### 2.5 改进输出说明
**文件**: `README.md`
- 增强 "Outputs" 部分
- 列出所有生成文件及其含义
- 说明 `runs/expoly-<timestamp>/` 目录结构

---

## 阶段 3: 测试 + CI

### 3.1 创建测试目录结构
**新建目录**: `tests/`
**新建文件**:
- `tests/__init__.py`
- `tests/conftest.py` - pytest 配置和 fixtures
- `tests/test_frames.py` - Frame 类测试
- `tests/test_carve.py` - Carve 功能测试
- `tests/test_polish.py` - Polish 功能测试
- `tests/test_cli.py` - CLI 测试
- `tests/fixtures/` - 测试数据目录

---

### 3.2 实现核心测试
**文件**: `tests/test_frames.py`
- 测试 HDF5 数据集查找
- 测试 Frame 初始化
- 测试 grain ID 查询

**文件**: `tests/test_carve.py`
- 测试确定性映射（相同输入→相同输出）
- 测试 FCC/BCC/DIA 晶格生成

**文件**: `tests/test_polish.py`
- 测试 LAMMPS 数据文件生成
- 测试输出文件存在性
- 测试原子计数和 header 字段

**文件**: `tests/test_cli.py`
- 测试 `expoly --help`
- 测试 `expoly run` 参数解析
- 测试错误处理

**验证**: `pytest tests/ -v`

---

### 3.3 添加测试依赖
**文件**: `pyproject.toml`
- 确保 `pytest` 在 `[project.optional-dependencies.dev]` 中
- 添加 `pytest-cov`（可选，用于覆盖率）

---

### 3.4 创建 GitHub Actions CI
**新建文件**: `.github/workflows/tests.yml`
- 在 Python 3.10 和 3.11 上运行测试
- 触发条件: push 和 PR
- 步骤: 安装依赖 → 运行 pytest

**验证**: 提交后检查 GitHub Actions 运行

---

### 3.5 添加代码格式化（最小化）
**文件**: `pyproject.toml`
- 添加 `[tool.ruff]` 配置（如果使用 ruff）
- 或添加 `[tool.black]` 和 `[tool.isort]`（如果使用 black+isort）
- 仅格式化新代码，避免大范围 diff

**验证**: `ruff check .` 或 `black --check .`

---

## 阶段 4: CLI 改进

### 4.1 改进帮助信息
**文件**: `src/expoly/cli.py`
- 使用 `argparse.ArgumentGroup` 组织选项
- 分组: "Input", "Carving", "Polish", "Output", "Advanced"
- 确保所有参数都有清晰的 `help` 文本

**验证**: `expoly run --help` 显示分组

---

### 4.2 实现 `expoly doctor` 子命令
**文件**: `src/expoly/cli.py`
- 在 `build_parser()` 中添加 `doctor` 子命令
- 创建 `doctor_command()` 函数
- 检查项:
  - Dream3D 文件存在性
  - HDF5 数据集路径有效性
  - 必需列存在性
  - 值范围合理性（HX/HY/HZ）
  - OVITO 可导入性
- 输出可操作的修复建议

**验证**: `expoly doctor --dream3d <file>` 运行并输出诊断

---

### 4.3 改进错误消息
**文件**: `src/expoly/cli.py`, `src/expoly/frames.py`, `src/expoly/polish.py`
- 将通用错误替换为具体错误消息
- 格式: "missing column X in HDF5 group Y; use --h5-grain-dset ..."
- 添加错误代码或类型（便于调试）

**验证**: 触发各种错误，检查消息清晰度

---

### 4.4 实现 `pipeline.run()` 函数
**文件**: `src/expoly/pipeline.py`
- 封装 `cli.run_noninteractive()` 逻辑
- 提供程序化 API（接受参数而非 argparse.Namespace）
- 保持与 CLI 行为一致

**验证**: `from expoly import run; run(...)` 可调用

---

## 阶段 5: 基准测试

### 5.1 创建基准测试目录
**新建目录**: `benchmarks/`
**新建文件**:
- `benchmarks/README.md` - 基准测试说明
- `benchmarks/benchmark.py` - 基准测试脚本
- `benchmarks/generate_toy_data.py` - 生成不同大小的测试数据

---

### 5.2 实现基准测试脚本
**文件**: `benchmarks/benchmark.py`
- 测试小/中规模数据转换时间
- 测量阶段: Frame 加载、Carve、Polish
- 输出 CSV 或 JSON 结果

**验证**: `python benchmarks/benchmark.py` 运行并生成结果

---

### 5.3 在文档中记录基准结果
**文件**: `README.md` 或 `docs/benchmarks.md`
- 添加基准测试结果表格
- 包含: 数据大小、转换时间、内存使用（如果可能）

**验证**: 表格存在且数据合理

---

## 实施顺序总结

```
阶段 1: 结构改进 (1-2 天)
  ├─ 1.1 清理依赖
  ├─ 1.2 创建 docs/
  ├─ 1.3 创建 examples/
  ├─ 1.4 添加元数据文件
  └─ 1.5 检查 LICENSE

阶段 2: README 改进 (1 天)
  ├─ 2.1 价值主张
  ├─ 2.2 快速开始
  ├─ 2.3 流程图
  ├─ 2.4 失败模式
  └─ 2.5 输出说明

阶段 3: 测试 + CI (2-3 天)
  ├─ 3.1 测试目录结构
  ├─ 3.2 核心测试实现
  ├─ 3.3 测试依赖
  ├─ 3.4 GitHub Actions
  └─ 3.5 代码格式化

阶段 4: CLI 改进 (1-2 天)
  ├─ 4.1 改进帮助信息
  ├─ 4.2 实现 doctor 命令
  ├─ 4.3 改进错误消息
  └─ 4.4 实现 pipeline.run()

阶段 5: 基准测试 (1 天)
  ├─ 5.1 基准测试目录
  ├─ 5.2 基准测试脚本
  └─ 5.3 文档记录
```

## 验证清单

每个阶段完成后运行:

```bash
# 1. 安装检查
pip install -e .
expoly --help

# 2. 测试
pytest tests/ -v

# 3. 示例运行
python examples/minimal_example.py

# 4. Doctor 检查
expoly doctor --dream3d <test_file>

# 5. CI 检查（提交后）
# 检查 GitHub Actions 是否通过
```

## 注意事项

1. **向后兼容**: 所有现有 CLI 参数必须继续工作
2. **小步提交**: 每个阶段完成后提交，便于审查
3. **测试优先**: 新功能先写测试，再实现
4. **文档同步**: 代码变更时同步更新文档
5. **性能**: 基准测试不应显著影响现有性能
