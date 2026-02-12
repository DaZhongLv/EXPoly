# EXPoly Quick Start Guide

## ✅ Verify Files

All files have been saved. Follow these steps to verify and start using the project.

## 📋 File Check

### Core Files (confirmed)

✅ **Project configuration**
- `pyproject.toml` (version 1.0.0)
- `LICENSE` (MIT)
- `.gitignore` (configured to exclude *.dream3d)

✅ **Source code** (src/expoly/)
- All 8 Python module files

✅ **Tests** (tests/)
- 4 test files, ~25 tests

✅ **Documentation**
- README.md (full, 328 lines)
- docs/use_cases.md
- docs/benchmarks.md

✅ **Examples and benchmarks**
- examples/ (2 example files)
- benchmarks/ (2 benchmark scripts)

✅ **CI/CD**
- .github/workflows/tests.yml

## 🚀 Get Started

### 1. Local install and verification

```bash
# Go to project directory
cd /Users/lvmeizhong/Desktop/expoly-with-legacy/EXPoly

# Install (editable mode)
pip install -e ".[dev]"
pip install ovito

# Verify installation
expoly --help

# Run tests
pytest tests/ -v

# Test doctor command
expoly doctor --dream3d An0new6.dream3d --hx 0:50 --hy 0:50 --hz 0:50
```

### 2. Run examples

**Option A: Download real sample data**
```bash
# From CMU Grain Boundary Data Archive
# Visit: http://mimp.materials.cmu.edu/~gr20/Grain_Boundary_Data_Archive/Ni_velocity/Ni_velocity.html
# Download "Microstructure Data" archive (367 MB, contains 6 Dream3D files)
# Extract and use any Dream3D file
```

**Option B: Use a local sample file (if present)**
```bash
cd examples
python minimal_example.py
# Will auto-detect and use An0new6.dream3d if present
```

**Option C: Generate test data**
```bash
# Generate a small test file
python examples/toy_data_generator.py

# Run example (uses generated toy_data.dream3d)
python examples/minimal_example.py
```

### 3. Use your own data

```bash
expoly run \
  --dream3d /path/to/your_data.dream3d \
  --hx 0:100 --hy 0:100 --hz 0:100 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524
```

## 📤 Upload to GitHub

### About sample file (An0new6.dream3d)

**Important**: This file is about **554 MB** and is too large to commit to GitHub.

**What’s in place**:
1. ✅ **Configured**: `.gitignore` excludes `*.dream3d` files
2. ✅ **Automatic**: Git will ignore the file and it will not be uploaded
3. ✅ **Alternative**: Users can use `toy_data_generator.py` to generate small test files

### Upload steps

```bash
# 1. Check Git status (confirm .dream3d is ignored)
git status
# You should not see An0new6.dream3d

# 2. Initialize repo (if not already)
git init
git branch -M main

# 3. Add files
git add .
git status  # Confirm no large files

# 4. Create initial commit
git commit -m "feat: v1.0.0 - Professional refactoring release

Complete repository structure with:
- Comprehensive documentation and examples
- Full test suite with pytest
- GitHub Actions CI workflow
- Improved CLI with doctor command
- Programmatic API (pipeline.run)
- Benchmarking infrastructure"

# 5. After creating the repo on GitHub, add remote and push
git remote add origin https://github.com/YOUR_USERNAME/EXPoly.git
git push -u origin main
```

### After uploading

Confirm that:
- ✅ All code files are present
- ✅ All documentation is present
- ✅ `.dream3d` files are **not** in the repo (this is correct)
- ✅ `.gitignore` exists

## 📝 File checklist

### Files to upload (< 5 MB total)

**Source code**:
- `src/expoly/*.py` (8 files)

**Tests**:
- `tests/*.py` (5 files)

**Documentation**:
- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `LICENSE`
- `docs/*.md` (2 files)

**Configuration**:
- `pyproject.toml`
- `.gitignore`
- `.github/workflows/tests.yml`

**Examples and benchmarks**:
- `examples/*.py` (2 files)
- `benchmarks/*.py` (2 files)

### Files that should not be uploaded (excluded)

- ❌ `An0new6.dream3d` (554 MB – too large)
- ❌ `__pycache__/` (Python cache)
- ❌ `*.egg-info/` (build artifacts)
- ❌ `.venv/` (virtual environment)
- ❌ `runs/` (output directory)

## 🎯 Next steps

1. ✅ **Verify locally**: Run tests and examples
2. ⏭️ **Initialize Git**: `git init` (if needed)
3. ⏭️ **Create GitHub repo**: Create a new repository on GitHub
4. ⏭️ **Push**: `git push`
5. ⏭️ **Sample data**:
   - Option A: Keep excluded (recommended; users provide their own data)
   - Option B: Use Git LFS if you need to include it
   - Option C: Host on GitHub Releases

## 📚 Related docs

- `GITHUB_SETUP.md` – GitHub setup guide
- `FILE_CHECKLIST.md` – Full file checklist
- `README.md` – Main documentation and usage
