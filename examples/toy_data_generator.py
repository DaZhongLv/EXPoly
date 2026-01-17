#!/usr/bin/env python3
"""
Generate a minimal synthetic Dream3D HDF5 file for testing EXPoly.

This creates a small voxel grid with a few grains and Euler angles.
"""

from __future__ import annotations

import h5py
import numpy as np
from pathlib import Path


def create_toy_dream3d(output_path: Path, size: tuple[int, int, int] = (20, 20, 20)) -> None:
    """
    Create a minimal Dream3D HDF5 file with:
    - FeatureIds: 3D array of grain IDs (0 = void, 1-N = grains)
    - EulerAngles: 3D array (z,y,x,3) of Euler angles (radians)
    - NumNeighbors: Per-grain neighbor counts
    - NeighborList: Flattened neighbor list
    - DIMENSIONS: [X, Y, Z] volume dimensions
    """
    z_size, y_size, x_size = size
    
    # Create a simple grain structure: 3 grains in a 20x20x20 volume
    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    
    # Grain 1: lower-left region
    feature_ids[0:10, 0:10, 0:10, 0] = 1
    
    # Grain 2: upper-right region
    feature_ids[10:20, 10:20, 10:20, 0] = 2
    
    # Grain 3: middle region
    feature_ids[5:15, 5:15, 5:15, 0] = 3
    
    # Euler angles (Bunge convention, radians)
    # Each grain gets a different orientation
    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    
    # Grain 1: orientation (0, 0, 0)
    euler_angles[0:10, 0:10, 0:10, :] = [0.0, 0.0, 0.0]
    
    # Grain 2: orientation (π/4, π/6, π/3)
    euler_angles[10:20, 10:20, 10:20, :] = [np.pi/4, np.pi/6, np.pi/3]
    
    # Grain 3: orientation (π/2, π/4, π/2)
    euler_angles[5:15, 5:15, 5:15, :] = [np.pi/2, np.pi/4, np.pi/2]
    
    # Simple neighbor list (each grain has 2 neighbors)
    num_neighbors = np.array([2, 2, 2], dtype=np.int32)  # Grain 1, 2, 3
    neighbor_list = np.array([2, 3, 1, 3, 1, 2], dtype=np.int32)  # Flattened
    
    # Dimensions
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    
    # Write HDF5 file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create typical Dream3D structure
        container = f.create_group('DataContainers')
        volume = container.create_group('SyntheticVolumeDataContainer')
        cell_data = volume.create_group('CellData')
        
        cell_data.create_dataset('FeatureIds', data=feature_ids)
        cell_data.create_dataset('EulerAngles', data=euler_angles)
        
        # Feature data (grain-level)
        feature_data = volume.create_group('CellFeatureData')
        feature_data.create_dataset('NumNeighbors', data=num_neighbors)
        feature_data.create_dataset('NeighborList', data=neighbor_list)
        
        # Geometry
        geometry = volume.create_group('_SIMPL_GEOMETRY')
        geometry.create_dataset('DIMENSIONS', data=dimensions)
    
    print(f"Created toy Dream3D file: {output_path}")
    print(f"  Volume size: {x_size} × {y_size} × {z_size}")
    print(f"  Grains: 1, 2, 3 (plus void=0)")
    print(f"  HDF5 structure: DataContainers/SyntheticVolumeDataContainer/...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate toy Dream3D HDF5 file')
    parser.add_argument('--output', type=Path, default=Path('toy_data.dream3d'),
                       help='Output HDF5 file path')
    parser.add_argument('--size', type=int, nargs=3, default=[20, 20, 20],
                       metavar=('Z', 'Y', 'X'),
                       help='Volume size (default: 20 20 20)')
    
    args = parser.parse_args()
    create_toy_dream3d(args.output, tuple(args.size))
