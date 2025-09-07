
import os
from pathlib import Path
import numpy as np
import h5py
import pytest

from expoly.frames import Frame

@pytest.fixture(scope="session")
def dream3d_path():
    p = os.getenv("EXPOLY_DREAM3D_PATH", "Alpoly_elongate.dream3d")
    p = Path(p)
    assert p.exists(), f"DREAM.3D file not found: {p} (set EXPOLY_DREAM3D_PATH)"
    return p

# export EXPOLY_DREAM3D_PATH="/Users/<你的用户名>/path/to/Alpoly_elongate.dream3d"

def test_frame_load_and_shapes(dream3d_path):
    # prefer_groups includes both CellData and Grain Data for robustness
    f = Frame(dream3d_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])  # noqa
    # Dimensions are stored as [X, Y, Z]
    assert (f.HX_lim, f.HY_lim, f.HZ_lim) == (200, 200, 400)

    # Neighbor list size must match sum of per-feature counts
    assert int(f.Num_NN.sum()) == len(f.Num_list)

def test_pick_a_valid_grain_and_basic_queries(dream3d_path):
    # Read one valid grain id from the HDF5 (small scan) to make the test robust
    with h5py.File(dream3d_path, "r") as h:
        fid = h["DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds"]
        zmax, ymax, xmax = fid.shape[:3]
        # Scan a small block to find a non-zero id
        block = fid[0:min(16,zmax), 0:min(16,ymax), 0:min(16,xmax), 0]
        uniq = np.unique(block)
        # pick the smallest positive id
        valid_ids = uniq[uniq > 0]
        assert valid_ids.size > 0, "No positive grain id found in sampled block"
        gid = int(valid_ids.min())

    f = Frame(dream3d_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])  # noqa

    # from_ID_to_D returns a non-empty dataframe with expected columns and ID
    df = f.from_ID_to_D(gid)
    assert not df.empty
    assert list(df.columns) == ['HZ','HY','HX','ID']
    assert (df['ID'] == gid).all()

    # get_grain_size >= len(from_ID_to_D) (depending on duplicates / shape flattening)
    assert f.get_grain_size(gid) >= len(df)

    # search_NN returns an array of integers and does not include itself
    nn = f.search_NN(gid)
    assert isinstance(nn, np.ndarray)
    assert nn.dtype.kind in ('i','u')
    assert gid not in nn.tolist()

def test_find_volume_grain_id_small_window(dream3d_path):
    f = Frame(dream3d_path, prefer_groups=["CellData","Grain Data","CellFeatureData","_SIMPL_GEOMETRY"])  # noqa
    gids = f.find_volume_grain_ID((0, 9), (0, 9), (0, 9))  # tiny 10x10x10 region
    assert isinstance(gids, np.ndarray)
    assert gids.size > 0
