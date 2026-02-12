#!/usr/bin/env python3
"""
Export Grain Point Cloud from HDF5 Data

Pure data extraction: reads voxel centers and grain IDs from HDF5,
extracts grain points and margin points, outputs as simple .xyz point cloud files.

No visualization, no voxelization, no rendering.

Usage:
    python export_grain_points.py --dream3d An0new6.dream3d --grain-id 111
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

# Add src to path to import expoly modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from expoly.frames import Frame


def print_h5_tree(h5_path: Path, max_depth: int = 10):
    """Print HDF5 file structure."""
    def _print_tree(name, obj, depth=0):
        if depth > max_depth:
            return
        indent = "  " * depth
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset: shape={obj.shape}, dtype={obj.dtype})")
        else:
            print(f"{indent}{name}/ (Group)")

    with h5py.File(h5_path, 'r') as f:
        print(f"\nHDF5 Tree for: {h5_path}")
        print("=" * 60)
        f.visititems(_print_tree)
        print("=" * 60)


def load_voxel_data(
    h5_path: Path,
    positions_path: Optional[str] = None,
    grain_id_path: Optional[str] = None,
    h5_euler_dset: Optional[str] = None,
    h5_numneighbors_dset: Optional[str] = None,
    h5_neighborlist_dset: Optional[str] = None,
    h5_dimensions_dset: Optional[str] = None,
    auto_detect: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load voxel centers and grain IDs from HDF5 file.

    Returns:
        (voxel_centers, grain_ids) where:
        - voxel_centers: (N, 3) array of [X, Y, Z] coordinates
        - grain_ids: (N,) array of grain IDs
    """
    # Use Frame class to load data (reuses existing logic)
    mapping = {
        "GrainId": grain_id_path or "FeatureIds",
        "Euler": h5_euler_dset or "EulerAngles",
        "Num_NN": h5_numneighbors_dset or "NumNeighbors",
        "Num_list": h5_neighborlist_dset or "NeighborList",
        "Dimension": h5_dimensions_dset or "DIMENSIONS",
    }

    frame = Frame(str(h5_path), mapping=mapping)

    # Get positions from frame (voxel centers)
    # Frame stores positions as HX, HY, HZ in the GrainId array indices
    grain_id_array = frame.GrainId
    positions = []
    grain_ids = []

    # Extract positions and grain IDs from the 3D grid
    if grain_id_array.ndim == 3:
        z_dim, y_dim, x_dim = grain_id_array.shape
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    grain_id = grain_id_array[z, y, x]
                    if grain_id > 0:  # Only non-zero grain IDs
                        positions.append([x, y, z])
                        grain_ids.append(grain_id)
    elif grain_id_array.ndim == 4:
        z_dim, y_dim, x_dim, _ = grain_id_array.shape
        for z in range(z_dim):
            for y in range(y_dim):
                for x in range(x_dim):
                    grain_id = grain_id_array[z, y, x, 0]
                    if grain_id > 0:
                        positions.append([x, y, z])
                        grain_ids.append(grain_id)

    positions = np.array(positions, dtype=float)
    grain_ids = np.array(grain_ids, dtype=int)

    print(f"Loaded {len(positions)} voxel centers")
    print(f"  - Positions shape: {positions.shape}")
    print(f"  - Grain IDs shape: {grain_ids.shape}")
    print(f"  - Unique grain IDs: {len(np.unique(grain_ids))}")

    return positions, grain_ids


def extract_grain_points(
    positions: np.ndarray,
    grain_ids: np.ndarray,
    target_grain_id: int,
) -> np.ndarray:
    """Extract point cloud for a specific grain."""
    mask = grain_ids == target_grain_id
    grain_points = positions[mask]
    print(f"Extracted {len(grain_points)} points for grain {target_grain_id}")
    return grain_points


def extract_margin_points(
    frame: Frame,
    grain_id: int,
) -> np.ndarray:
    """
    Extract margin points using Frame.find_grain_NN_with_out() logic.

    Returns margin voxel centers as point cloud.
    """
    # Get margin data from frame
    margin_df = frame.find_grain_NN_with_out(grain_id)

    # Extract margin points (margin-ID == 1 or 2)
    margin_mask = (margin_df['margin-ID'] == 1) | (margin_df['margin-ID'] == 2)
    margin_df_filtered = margin_df[margin_mask]

    # Get positions (HX, HY, HZ)
    margin_points = margin_df_filtered[['HX', 'HY', 'HZ']].to_numpy()

    print(f"Extracted {len(margin_points)} margin points for grain {grain_id}")

    return margin_points


def save_point_cloud_ply(points: np.ndarray, output_path: Path):
    """
    Save point cloud to PLY file (Blender-compatible).

    Format: PLY point cloud, plain text format.
    """
    if len(points) == 0:
        print(f"Warning: No points to save to {output_path}")
        return

    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write points
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Saved {len(points)} points to PLY: {output_path}")


def save_point_cloud_xyz(points: np.ndarray, output_path: Path):
    """
    Save point cloud to .xyz file (simple text format).

    Format: plain text, one point per line: x y z
    """
    if len(points) == 0:
        print(f"Warning: No points to save to {output_path}")
        return

    with open(output_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"Saved {len(points)} points to XYZ: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Grain Point Cloud from HDF5 Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--dream3d",
        type=str,
        required=True,
        help="Path to Dream3D HDF5 file"
    )
    parser.add_argument(
        "--grain-id",
        type=int,
        required=True,
        help="Target grain ID to extract"
    )

    # HDF5 paths
    parser.add_argument(
        "--print-h5-tree",
        action="store_true",
        help="Print HDF5 file structure and exit"
    )
    parser.add_argument(
        "--h5-grain-dset",
        type=str,
        default=None,
        help="HDF5 dataset name for grain IDs (default: FeatureIds)"
    )
    parser.add_argument(
        "--h5-euler-dset",
        type=str,
        default=None,
        help="HDF5 dataset name for Euler angles (default: EulerAngles)"
    )
    parser.add_argument(
        "--h5-numneighbors-dset",
        type=str,
        default=None,
        help="HDF5 dataset name for number of neighbors (default: NumNeighbors)"
    )
    parser.add_argument(
        "--h5-neighborlist-dset",
        type=str,
        default=None,
        help="HDF5 dataset name for neighbor list (default: NeighborList)"
    )
    parser.add_argument(
        "--h5-dimensions-dset",
        type=str,
        default=None,
        help="HDF5 dataset name for dimensions (default: DIMENSIONS)"
    )

    # Output
    parser.add_argument(
        "--output-grain",
        type=str,
        default=None,
        help="Output file for grain points (default: grain_<ID>_points.ply)"
    )
    parser.add_argument(
        "--output-margin",
        type=str,
        default=None,
        help="Output file for margin points (default: grain_<ID>_margin_points.ply)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=['ply', 'xyz'],
        default='ply',
        help="Output format: 'ply' for Blender (default), 'xyz' for simple text"
    )

    args = parser.parse_args()

    h5_path = Path(args.dream3d)
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        sys.exit(1)

    # Print HDF5 tree if requested
    if args.print_h5_tree:
        print_h5_tree(h5_path)
        sys.exit(0)

    # Load data using Frame
    print("Loading data from HDF5...")
    mapping = {
        "GrainId": args.h5_grain_dset or "FeatureIds",
        "Euler": args.h5_euler_dset or "EulerAngles",
        "Num_NN": args.h5_numneighbors_dset or "NumNeighbors",
        "Num_list": args.h5_neighborlist_dset or "NeighborList",
        "Dimension": args.h5_dimensions_dset or "DIMENSIONS",
    }

    try:
        frame = Frame(str(h5_path), mapping=mapping)
    except Exception as e:
        print(f"Error loading Frame: {e}")
        print("\nTip: Use --print-h5-tree to inspect the HDF5 structure")
        sys.exit(1)

    # Extract grain points
    print(f"\nExtracting grain {args.grain_id} points...")
    grain_df = frame.from_ID_to_D(args.grain_id)
    grain_points = grain_df[['HX', 'HY', 'HZ']].to_numpy()

    if len(grain_points) == 0:
        print(f"Error: No points found for grain {args.grain_id}")
        sys.exit(1)

    print(f"Extracted {len(grain_points)} grain points")

    # Extract margin points using Frame logic
    print("\nExtracting margin points...")
    margin_points = extract_margin_points(frame, args.grain_id)

    # Determine output paths
    ext = args.format
    if args.output_grain is None:
        output_grain = Path(f"grain_{args.grain_id}_points.{ext}")
    else:
        output_grain = Path(args.output_grain)

    if args.output_margin is None:
        output_margin = Path(f"grain_{args.grain_id}_margin_points.{ext}")
    else:
        output_margin = Path(args.output_margin)

    # Save point clouds
    print(f"\nSaving point clouds (format: {args.format})...")
    if args.format == 'ply':
        save_point_cloud_ply(grain_points, output_grain)
        save_point_cloud_ply(margin_points, output_margin)
    else:
        save_point_cloud_xyz(grain_points, output_grain)
        save_point_cloud_xyz(margin_points, output_margin)

    print("\nDone!")


if __name__ == "__main__":
    main()
