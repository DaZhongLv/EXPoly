#!/usr/bin/env python3
"""
Generate synthetic Dream3D HDF5 files of different sizes for benchmarking.

This script creates multiple test files with varying volumes and grain counts.
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
from pathlib import Path


def create_benchmark_dream3d(
    output_path: Path,
    size: tuple[int, int, int],
    n_grains: int = 5,
) -> None:
    """
    Create a Dream3D HDF5 file for benchmarking.
    
    Parameters
    ----------
    output_path : Path
        Output file path
    size : tuple[int, int, int]
        Volume size (z, y, x)
    n_grains : int
        Number of grains to create
    """
    z_size, y_size, x_size = size
    
    # Create grain structure: distribute grains evenly
    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    
    # Simple distribution: divide volume into n_grains regions
    grain_size_per_dim = max(1, min(z_size, y_size, x_size) // n_grains)
    
    for gid in range(1, n_grains + 1):
        # Create a region for each grain
        z_start = (gid - 1) * grain_size_per_dim
        z_end = min(z_start + grain_size_per_dim, z_size)
        y_start = (gid - 1) * grain_size_per_dim
        y_end = min(y_start + grain_size_per_dim, y_size)
        x_start = (gid - 1) * grain_size_per_dim
        x_end = min(x_start + grain_size_per_dim, x_size)
        
        if z_start < z_size and y_start < y_size and x_start < x_size:
            feature_ids[z_start:z_end, y_start:y_end, x_start:x_end, 0] = gid
    
    # Euler angles (Bunge convention, radians)
    # Each grain gets a different orientation
    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    for gid in range(1, n_grains + 1):
        mask = (feature_ids[:, :, :, 0] == gid)
        if np.any(mask):
            # Assign random-ish orientation
            phi1 = (gid * np.pi / 4) % (2 * np.pi)
            Phi = (gid * np.pi / 6) % np.pi
            phi2 = (gid * np.pi / 3) % (2 * np.pi)
            euler_angles[mask] = [phi1, Phi, phi2]
    
    # Neighbor list (simplified: each grain has 2 neighbors)
    num_neighbors = np.array([2] * n_grains, dtype=np.int32)
    neighbor_list = []
    for i in range(1, n_grains + 1):
        # Connect to next and previous grain
        neighbor_list.extend([
            (i % n_grains) + 1,
            ((i - 2) % n_grains) + 1,
        ])
    neighbor_list = np.array(neighbor_list, dtype=np.int32)
    
    # Dimensions
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    
    # Write HDF5 file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
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
    
    print(f"Created: {output_path}")
    print(f"  Size: {x_size} × {y_size} × {z_size} = {x_size * y_size * z_size:,} voxels")
    print(f"  Grains: {n_grains}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark Dream3D HDF5 files'
    )
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[20, 50, 100],
        help='Volume sizes to generate (default: 20 50 100). '
             'Each size creates a cube of that dimension.'
    )
    parser.add_argument(
        '--grains',
        type=int,
        default=5,
        help='Number of grains per file (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('benchmarks/fixtures'),
        help='Output directory (default: benchmarks/fixtures)'
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for size in args.sizes:
        output_path = args.output_dir / f"benchmark_{size}x{size}x{size}.dream3d"
        create_benchmark_dream3d(
            output_path,
            size=(size, size, size),
            n_grains=args.grains,
        )


if __name__ == '__main__':
    main()
