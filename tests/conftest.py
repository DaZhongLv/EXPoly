"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from expoly.frames import Frame


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test outputs."""
    return tmp_path


@pytest.fixture
def toy_dream3d_file(tmp_dir: Path) -> Path:
    """
    Create a minimal Dream3D HDF5 file for testing.
    Returns path to the created file.
    """
    h5_path = tmp_dir / "toy.dream3d"

    z_size, y_size, x_size = 20, 20, 20

    # Create simple grain structure: 2 grains
    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    feature_ids[0:10, 0:10, 0:10, 0] = 1
    feature_ids[10:20, 10:20, 10:20, 0] = 2

    # Euler angles (Bunge, radians)
    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    euler_angles[0:10, 0:10, 0:10, :] = [0.0, 0.0, 0.0]
    euler_angles[10:20, 10:20, 10:20, :] = [np.pi/4, np.pi/6, np.pi/3]

    # Neighbor list (simple: each grain has 1 neighbor)
    num_neighbors = np.array([1, 1], dtype=np.int32)
    neighbor_list = np.array([2, 1], dtype=np.int32)

    # Dimensions
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)

    with h5py.File(h5_path, 'w') as f:
        container = f.create_group('DataContainers')
        volume = container.create_group('SyntheticVolumeDataContainer')
        cell_data = volume.create_group('CellData')
        cell_data.create_dataset('FeatureIds', data=feature_ids)
        cell_data.create_dataset('EulerAngles', data=euler_angles)

        feature_data = volume.create_group('CellFeatureData')
        feature_data.create_dataset('NumNeighbors', data=num_neighbors)
        feature_data.create_dataset('NeighborList', data=neighbor_list)

        geometry = volume.create_group('_SIMPL_GEOMETRY')
        geometry.create_dataset('DIMENSIONS', data=dimensions)

    return h5_path


@pytest.fixture
def frame_instance(toy_dream3d_file: Path) -> Frame:
    """Create a Frame instance from toy data."""
    return Frame(str(toy_dream3d_file))
