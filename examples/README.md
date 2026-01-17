# EXPoly Examples

This directory contains runnable examples demonstrating EXPoly usage.

## Quick Start

1. **Minimal Example** (`minimal_example.py`):
   - Generates a small synthetic Dream3D HDF5 file
   - Runs the full carve + polish pipeline
   - Demonstrates basic usage

2. **Toy Data Generator** (`toy_data_generator.py`):
   - Creates synthetic voxel data for testing
   - Useful for development and benchmarking

## Running Examples

```bash
# From the repository root
cd examples
python minimal_example.py
```

## Example Output

Examples will create output in `runs/expoly-<timestamp>/`:
- `raw_points.csv`: Carved atoms
- `final.data`: Clean LAMMPS data file

## Notes

- Examples automatically use `An0new6.dream3d` if available in the repository root
- Otherwise, examples generate synthetic data (small volumes, few grains)
- Real-world usage will require actual Dream3D HDF5 files
- Adjust parameters (lattice type, ratio, etc.) as needed for your use case

## Sample Data

**Download sample data**:
- **CMU Grain Boundary Data Archive**: [Nickel Grain Boundary Velocity Data](http://mimp.materials.cmu.edu/~gr20/Grain_Boundary_Data_Archive/Ni_velocity/Ni_velocity.html)
  - Download the "Microstructure Data" archive (367 MB, contains 6 Dream3D files)
  - These are real experimental data from Ni polycrystals
  - Extract and use any Dream3D file with EXPoly

**Local sample**:
- If you have `An0new6.dream3d` locally, the examples will automatically use it
- Otherwise, use the toy data generator or download from the link above
