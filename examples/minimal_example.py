#!/usr/bin/env python3
"""
Minimal example demonstrating EXPoly usage.

This example:
1. Generates a small synthetic Dream3D HDF5 file
2. Runs the full carve + polish pipeline
3. Shows how to use the CLI programmatically
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Add parent directory to path to import toy_data_generator
sys.path.insert(0, str(Path(__file__).parent))

from toy_data_generator import create_toy_dream3d


def main():
    """Run minimal EXPoly example."""
    # Check if real data exists, otherwise generate toy data
    repo_root = Path(__file__).parent.parent
    real_data = repo_root / "An0new6.dream3d"
    toy_data = Path(__file__).parent / "toy_data.dream3d"
    
    if real_data.exists():
        print("=" * 60)
        print("Using real data: An0new6.dream3d")
        print("=" * 60)
        dream3d_path = real_data
    else:
        # 1. Generate toy data
        print("=" * 60)
        print("Step 1: Generating toy Dream3D HDF5 file...")
        print("=" * 60)
        create_toy_dream3d(toy_data, size=(20, 20, 20))
        dream3d_path = toy_data
    
    # 2. Run EXPoly
    print("\n" + "=" * 60)
    print("Step 2: Running EXPoly carve + polish pipeline...")
    print("=" * 60)
    
    # Example command (adjust parameters as needed)
    # For real data, use appropriate H ranges; for toy data, use 0:20
    if real_data.exists():
        # Conservative ranges for real data (adjust based on actual dimensions)
        hx_range, hy_range, hz_range = "0:100", "0:100", "0:100"
    else:
        hx_range, hy_range, hz_range = "0:20", "0:20", "0:20"
    
    # Build command with all HDF5 dataset names explicitly specified
    # For An0new6.dream3d: NeighborList2 (not default NeighborList)
    cmd = [
        "expoly", "run",
        "--dream3d", str(dream3d_path.absolute()),
        "--hx", hx_range,
        "--hy", hy_range,
        "--hz", hz_range,
        "--lattice", "FCC",
        "--ratio", "1.5",
        "--lattice-constant", "3.524",  # Ni
        "--h5-grain-dset", "FeatureIds",
        "--h5-euler-dset", "EulerAngles",
        "--h5-numneighbors-dset", "NumNeighbors",
        "--h5-neighborlist-dset", "NeighborList2",  # An0new6.dream3d uses NeighborList2
        "--h5-dimensions-dset", "DIMENSIONS",
        "--workers", "2",
        "--ovito-cutoff", "1.6",
        "--keep-tmp",  # Keep intermediate files for inspection
    ]
    
    # For toy data, use default dataset names (no --h5-*-dset needed)
    if not real_data.exists():
        # Remove h5 dataset arguments for toy data (uses defaults)
        cmd = [arg for arg in cmd if not arg.startswith("--h5-")]
        # Re-add only the essential arguments
        cmd = [
            "expoly", "run",
            "--dream3d", str(dream3d_path.absolute()),
            "--hx", hx_range,
            "--hy", hy_range,
            "--hz", hz_range,
            "--lattice", "FCC",
            "--ratio", "1.5",
            "--lattice-constant", "3.524",
            "--workers", "2",
            "--ovito-cutoff", "1.6",
            "--keep-tmp",
        ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 60)
        print("✓ EXPoly pipeline completed successfully!")
        print("=" * 60)
        print("\nCheck the output in: runs/expoly-<timestamp>/")
        print("  - raw_points.csv: All carved atoms")
        print("  - final.data: Clean LAMMPS data file")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ EXPoly failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("  1. Ensure EXPoly is installed: pip install -e .")
        print("  2. Ensure OVITO is installed: pip install ovito")
        print("  3. Check that the toy data file was created correctly")
        return 1
    except FileNotFoundError:
        print("\n✗ 'expoly' command not found.")
        print("  Install EXPoly first: pip install -e .")
        return 1


if __name__ == '__main__':
    sys.exit(main())
