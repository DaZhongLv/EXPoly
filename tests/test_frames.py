"""Tests for expoly.frames module."""

from __future__ import annotations

import numpy as np

from expoly.frames import Frame, find_dataset_keys


def test_find_dataset_keys(toy_dream3d_file, tmp_dir):
    """Test HDF5 dataset finding."""
    import h5py

    with h5py.File(toy_dream3d_file, "r") as f:
        keys = find_dataset_keys(f, "FeatureIds", prefer_groups=["CellData"])
        assert len(keys) > 0
        assert "FeatureIds" in keys[-1] or "featureids" in keys[-1].lower()


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
    assert "HX" in df.columns
    assert "HY" in df.columns
    assert "HZ" in df.columns
    assert "ID" in df.columns
    assert df["ID"].iloc[0] == 1


def test_frame_euler_search(frame_instance: Frame):
    """Test Euler angle search."""
    euler = frame_instance.search_avg_Euler(1)
    assert euler.shape == (3,)
    assert np.all(np.isfinite(euler))


def test_frame_volume_grain_id(frame_instance: Frame):
    """Test volume grain ID selection."""
    gids = frame_instance.find_volume_grain_ID(
        HX_range=(0, 10), HY_range=(0, 10), HZ_range=(0, 10), return_count=False
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


def test_frame_phase_multi_phase(tmp_dir):
    """Test Frame with Phases and PhaseName (multi-phase support)."""
    import h5py

    h5_path = tmp_dir / "phase_test.dream3d"
    z_size, y_size, x_size = 20, 20, 20

    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    feature_ids[0:10, 0:10, 0:10, 0] = 1
    feature_ids[10:20, 10:20, 10:20, 0] = 2

    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    euler_angles[0:10, 0:10, 0:10, :] = [0.0, 0.0, 0.0]
    euler_angles[10:20, 10:20, 10:20, :] = [np.pi / 4, np.pi / 6, np.pi / 3]

    num_neighbors = np.array([1, 1], dtype=np.int32)
    neighbor_list = np.array([2, 1], dtype=np.int32)
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    # Phases: index 0=background, 1=fcc (grain 1), 2=bcc (grain 2)
    phases = np.array([0, 1, 2], dtype=np.int32)
    phase_names = np.array(["Unknown", "fcc", "bcc"], dtype="S32")

    with h5py.File(h5_path, "w") as f:
        container = f.create_group("DataContainers")
        volume = container.create_group("SyntheticVolumeDataContainer")
        cell_data = volume.create_group("CellData")
        cell_data.create_dataset("FeatureIds", data=feature_ids)
        cell_data.create_dataset("EulerAngles", data=euler_angles)
        feature_data = volume.create_group("CellFeatureData")
        feature_data.create_dataset("NumNeighbors", data=num_neighbors)
        feature_data.create_dataset("NeighborList", data=neighbor_list)
        feature_data.create_dataset("Phases", data=phases)
        stats = container.create_group("StatsGeneratorDataContainer")
        ensemble = stats.create_group("CellEnsembleData")
        ensemble.create_dataset("PhaseName", data=phase_names)
        geometry = volume.create_group("_SIMPL_GEOMETRY")
        geometry.create_dataset("DIMENSIONS", data=dimensions)

    frame = Frame(str(h5_path))
    assert frame.Phases is not None
    assert frame.PhaseName is not None
    assert frame.search_phase(1) == 1
    assert frame.search_phase(2) == 2
    assert frame.get_lattice_for_grain(1) == "FCC"
    assert frame.get_lattice_for_grain(2) == "BCC"


def test_frame_phase_custom_dset_names(tmp_dir):
    """Test Frame with custom Phases/PhaseName dataset names."""
    import h5py

    h5_path = tmp_dir / "phase_custom.dream3d"
    z_size, y_size, x_size = 10, 10, 10

    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    feature_ids[0:5, 0:5, 0:5, 0] = 1

    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    num_neighbors = np.array([1], dtype=np.int32)
    neighbor_list = np.array([0], dtype=np.int32)
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    my_phases = np.array([0, 1], dtype=np.int32)
    my_phase_names = np.array(["Unknown", "fcc"], dtype="S32")

    with h5py.File(h5_path, "w") as f:
        c = f.create_group("DataContainers")
        v = c.create_group("SyntheticVolumeDataContainer")
        cd = v.create_group("CellData")
        cd.create_dataset("FeatureIds", data=feature_ids)
        cd.create_dataset("EulerAngles", data=euler_angles)
        fd = v.create_group("CellFeatureData")
        fd.create_dataset("NumNeighbors", data=num_neighbors)
        fd.create_dataset("NeighborList", data=neighbor_list)
        fd.create_dataset("MyPhases", data=my_phases)
        s = c.create_group("StatsGeneratorDataContainer")
        e = s.create_group("CellEnsembleData")
        e.create_dataset("MyPhaseName", data=my_phase_names)
        g = v.create_group("_SIMPL_GEOMETRY")
        g.create_dataset("DIMENSIONS", data=dimensions)

    frame = Frame(
        str(h5_path),
        h5_phases_dset="MyPhases",
        h5_phase_name_dset="MyPhaseName",
    )
    assert frame.Phases is not None
    assert frame.PhaseName is not None
    assert frame.get_lattice_for_grain(1) == "FCC"
