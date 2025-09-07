# EXPoly pytest for Alpoly_elongate.dream3d

This test targets the real DREAM.3D file you provided and exercises the `expoly.frames.Frame` loader
and a few light-weight queries.

## How to use

1. Place `Alpoly_elongate.dream3d` at your project root (same folder as `pyproject.toml`), or set:
   ```bash
   export EXPOLY_DREAM3D_PATH=/absolute/path/to/Alpoly_elongate.dream3d
   ```

2. Install your package in editable mode (with dev deps):
   ```bash
   pip install -e '.[dev]'
   ```

3. Run pytest:
   ```bash
   pytest -q
   ```

If your file lives elsewhere or your test runner uses a different CWD, use the environment variable.