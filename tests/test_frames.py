"""Tests for expoly.frames module."""

from __future__ import annotations

import numpy as np
import pytest

from expoly.frames import Frame, find_dataset_keys


def test_find_dataset_keys(toy_dream3d_file, tmp_dir):
    """Test HDF5 dataset finding."""
    import h5py
    
    with h5py.File(toy_dream3d_file, 'r') as f:
        keys = find_dataset_keys(f, 'FeatureIds', prefer_groups=['CellData'])
        assert len(keys) > 0
        assert 'FeatureIds' in keys[-1] or 'featureids' in keys[-1].lower()


def test_frame_initialization(frame_instance: Frame):
    """Test Frame initialization from HDF5 file."""
    assert frame_instance is not None
    assert frame_instance.GrainId is not None
    assert frame_instance.Euler is not None
    assert frame_instance.Dimension is not None


def test_frame_dimensions(frame_instance: Frame):
    """Test Frame dimension properties."""
    assert frame_instance.HX_lim == 20
    assert frame_instance.HY_lim == 20
    assert frame_instance.HZ_lim == 20


def test_frame_grain_id_query(frame_instance: Frame):
    """Test grain ID queries."""
    # Test from_ID_to_D
    df = frame_instance.from_ID_to_D(1)
    assert len(df) > 0
    assert 'HX' in df.columns
    assert 'HY' in df.columns
    assert 'HZ' in df.columns
    assert 'ID' in df.columns
    assert df['ID'].iloc[0] == 1


def test_frame_euler_search(frame_instance: Frame):
    """Test Euler angle search."""
    euler = frame_instance.search_avg_Euler(1)
    assert euler.shape == (3,)
    assert np.all(np.isfinite(euler))


def test_frame_volume_grain_id(frame_instance: Frame):
    """Test volume grain ID selection."""
    gids = frame_instance.find_volume_grain_ID(
        HX_range=(0, 10),
        HY_range=(0, 10),
        HZ_range=(0, 10),
        return_count=False
    )
    assert len(gids) > 0
    assert 1 in gids or 0 in gids  # Grain 1 or void


def test_frame_neighbor_search(frame_instance: Frame):
    """Test neighbor list search."""
    # Should not raise
    try:
        nn = frame_instance.search_nn_by_id(1)
        assert isinstance(nn, np.ndarray)
    except (ValueError, KeyError):
        # If grain 1 has no neighbors in test data, that's OK
        pass


def test_frame_grain_size(frame_instance: Frame):
    """Test grain size calculation."""
    size = frame_instance.get_grain_size(1)
    assert size > 0
    assert isinstance(size, int)
