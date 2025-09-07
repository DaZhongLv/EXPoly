# -*- coding: utf-8 -*-
from __future__ import annotations
import os, glob
from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption("--frame-path", action="store", default=None,
                     help="Path to frame data (.dream3d/.h5)")
    parser.addoption("--grain-id", action="store", type=int, default=None,
                     help="Grain ID to visualize")
    parser.addoption("--mode", action="store", default="both",
                     choices=["scatter", "voxels", "both"],
                     help="Visualization mode(s)")
    parser.addoption("--downsample", action="store", type=int, default=1,
                     help="Downsample factor for scatter (>=1)")
    parser.addoption("--show-margin", action="store_true", default=False,
                     help="Color by margin-ID")
    parser.addoption("--artifacts-dir", action="store", default="test_artifacts",
                     help="Directory (relative to repo root) to save images")

def _repo_root() -> Path:
    # 以本文件为基准，向上两级即仓库根（.../tests/conftest.py -> repo_root）
    return Path(__file__).resolve().parents[1]

def _auto_discover_frame() -> Path | None:
    root = _repo_root()
    pats = [
        str(root / "tests" / "data" / "*.dream3d"),
        str(root / "tests" / "data" / "*.h5"),
        str(root / "data" / "*.dream3d"),
        str(root / "data" / "*.h5"),
        str(root / "*.dream3d"),
        str(root / "*.h5"),
        str(root / "*" / "*.dream3d"),
        str(root / "*" / "*.h5"),
    ]
    for pat in pats:
        m = sorted(glob.glob(pat))
        if m:
            return Path(m[0]).resolve()
    return None

@pytest.fixture(scope="session")
def frame_path(request) -> Path:
    p = request.config.getoption("--frame-path") or os.getenv("FRAME_PATH") or os.getenv("DREAM3D_FRAME")
    if not p:
        auto_p = _auto_discover_frame()
        if auto_p:
            return auto_p
        pytest.skip("No frame file found. Provide --frame-path or set FRAME_PATH.")
    p = Path(p).expanduser().resolve()
    if not p.exists():
        pytest.skip(f"Frame file not found: {p}")
    return p

@pytest.fixture(scope="session")
def grain_id(request) -> int:
    gid = request.config.getoption("--grain-id")
    if gid is None:
        gid = int(os.getenv("GRAIN_ID", "1"))
    return gid

@pytest.fixture(scope="session")
def artifacts_dir(request) -> Path:
    # 把相对路径锚定到仓库根目录，避免从 tests/ 里起 pytest 时存到 tests/test_artifacts
    root = _repo_root()
    outdir = request.config.getoption("--artifacts-dir") or "test_artifacts"
    outpath = Path(outdir)
    if not outpath.is_absolute():
        outpath = root / outpath
    outpath = outpath.expanduser().resolve()
    outpath.mkdir(parents=True, exist_ok=True)
    return outpath

@pytest.fixture(scope="session")
def vis_mode(request) -> list[str]:
    m = request.config.getoption("--mode")
    return ["scatter", "voxels"] if m == "both" else [m]

@pytest.fixture(scope="session")
def downsample(request) -> int:
    return max(1, int(request.config.getoption("--downsample")))

@pytest.fixture(scope="session")
def show_margin(request) -> bool:
    return bool(request.config.getoption("--show-margin"))

def pytest_report_header(config):
    return [
        f"grain-id={config.getoption('--grain-id') or os.getenv('GRAIN_ID') or 1}",
        f"mode={config.getoption('--mode')}",
        f"artifacts-dir={config.getoption('--artifacts-dir') or 'test_artifacts (repo root)'}",
    ]
