import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import h5py

from expoly.frames import Frame
from expoly.carve import CarveConfig, prepare_carve, carve_points, process

DREAM_ENV = "EXPOLY_DREAM3D_PATH"

def _get_dream_path():
    p = os.getenv(DREAM_ENV, "Alpoly_elongate.dream3d")
    return Path(p)

dream_path = _get_dream_path()

pytestmark = pytest.mark.skipif(
    not dream_path.exists(),
    reason=f"DREAM.3D not found: set {DREAM_ENV} or place 'Alpoly_elongate.dream3d' at project root."
)

def _pick_valid_grain_id(p: Path) -> int:
    with h5py.File(p, "r") as h:
        fid = h["DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds"]
        zmax, ymax, xmax = fid.shape[:3]
        block = fid[0:min(16,zmax), 0:min(16,ymax), 0:min(16,xmax), 0]
        uniq = np.unique(block)
        valid = uniq[uniq > 0]
        assert valid.size > 0, "No positive grain id found in sampled block"
        return int(valid.min())

def test_prepare_and_carve_shapes():
    gid = _pick_valid_grain_id(dream_path)
    frame = Frame(dream_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])

    out_df = frame.from_ID_to_D(gid).copy()
    out_df["ID"] = gid

    cfg = CarveConfig(lattice="FCC", ratio=1.5, random_center=False, rng_seed=0, ci_radius=2**0.5)
    all_pts = prepare_carve(out_df, frame, cfg)
    assert all_pts.ndim == 2 and all_pts.shape[1] == 3 and all_pts.shape[0] > 0

    kept = carve_points(out_df, frame, cfg)
    assert kept.ndim == 2 and kept.shape[1] == 3
    assert kept.shape[0] <= all_pts.shape[0]

def test_process_determinism():
    gid = _pick_valid_grain_id(dream_path)
    frame = Frame(dream_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])

    cfg = CarveConfig(lattice="FCC", ratio=1.5, random_center=False, rng_seed=123, ci_radius=2**0.5)
    df1 = process(gid, frame, cfg)
    df2 = process(gid, frame, cfg)

    # Same seed & no random center -> deterministic
    pd.testing.assert_frame_equal(
        df1.sort_values(df1.columns.tolist()).reset_index(drop=True),
        df2.sort_values(df2.columns.tolist()).reset_index(drop=True)
    )

def test_process_columns():
    gid = _pick_valid_grain_id(dream_path)
    frame = Frame(dream_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])
    cfg = CarveConfig(lattice="FCC", ratio=1.5, random_center=False, rng_seed=0, ci_radius=2**0.5)
    df = process(gid, frame, cfg)
    required = {"X","Y","Z","HX","HY","HZ","margin-ID","grain-ID"}
    assert required.issubset(df.columns)
