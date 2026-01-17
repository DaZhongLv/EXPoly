# EXPoly Use Cases and Assumptions

## What EXPoly Solves

EXPoly converts experimental microstructure voxel data (from Dream3D HDF5 files) into MD-ready atomistic structures (LAMMPS data files). It is designed for researchers who need to:

1. **Reconstruct atomistic lattices** from voxel-based grain data
2. **Merge multiple grains** into a single simulation-ready structure
3. **Remove overlapping atoms** at grain boundaries using OVITO
4. **Export clean LAMMPS data files** for molecular dynamics simulations

## Key Assumptions

### Input Data Format

- **Dream3D HDF5 files** with specific dataset structure:
  - `FeatureIds` (or custom grain-ID dataset): 3D/4D array of grain IDs
  - `EulerAngles`: 3D array (z,y,x,3) of Euler angles (Bunge convention, radians)
  - `NumNeighbors` and `NeighborList`: For grain boundary detection
  - `DIMENSIONS`: Volume dimensions [X, Y, Z]

### Lattice Types

- Currently supports **cubic crystal families only**:
  - FCC (Face-Centered Cubic)
  - BCC (Body-Centered Cubic)
  - DIA (Diamond cubic)
- Each grain is assumed to have a **single, uniform orientation** (average Euler angles)

### Physical Assumptions

- **Lattice constant** must be known and provided (e.g., 3.524 Å for Ni)
- **Voxel-to-atom spacing ratio** (`--ratio`) determines grain size:
  - `ratio = 1`: One unit cell per voxel
  - `ratio = 2`: One unit cell spans 2×2×2 voxels
  - Larger ratio → smaller grain size in atomistic representation

### Computational Assumptions

- **OVITO is mandatory** for the polish step (overlap removal)
- **Memory usage** scales with:
  - Number of grains
  - Voxel volume (HX × HY × HZ ranges)
  - Lattice density (ratio parameter)
- **Processing time** scales with grain count and can be parallelized (`--workers`)

## Limitations

### Current Limitations

1. **Crystal structure**: Only cubic families (FCC/BCC/DIA) supported
2. **Single orientation per grain**: No sub-grain orientation variation
3. **Memory constraints**: Large volumes may require significant RAM
4. **OVITO dependency**: Cannot run polish step without OVITO installed
5. **Deterministic carving**: Random seed only affects ball center placement, not lattice structure

### Known Failure Modes

1. **Missing HDF5 datasets**: If `FeatureIds` or `EulerAngles` are not found, the tool will fail with a clear error message
2. **Invalid H ranges**: HX/HY/HZ ranges outside the volume dimensions will result in empty grain selection
3. **OVITO cutoff too large**: If `--ovito-cutoff` exceeds nearest-neighbor distance, legitimate atoms may be deleted
4. **Large file I/O**: Very large voxel grids may cause slow HDF5 reads

## Typical Workflow

1. **Prepare input**: Ensure Dream3D HDF5 file has required datasets
2. **Select region**: Choose HX/HY/HZ ranges to crop the volume
3. **Configure lattice**: Select lattice type (FCC/BCC/DIA) and ratio
4. **Run carve**: Generate atomistic points for each grain
5. **Run polish**: Remove overlaps and generate final LAMMPS data
6. **Validate output**: Check `final.data` in OVITO or LAMMPS

## When to Use EXPoly

✅ **Good fit for**:
- Converting EBSD/Dream3D voxel data to MD structures
- Multi-grain polycrystalline systems
- Systems with known lattice constants
- Cubic crystal structures

❌ **Not suitable for**:
- Non-cubic crystal structures (hexagonal, tetragonal, etc.)
- Systems requiring sub-grain orientation gradients
- Very large volumes without sufficient memory
- Real-time or streaming data processing

## Output Interpretation

- **`raw_points.csv`**: All carved atoms before overlap removal (for debugging)
- **`tmp_polish.in.data`**: Pre-OVITO LAMMPS data (can be opened in OVITO)
- **`ovito_cleaned.data`**: OVITO-processed data (intermediate)
- **`final.data`**: Final clean LAMMPS data file (recommended for simulations)
- **`final.dump`**: Optional dump format with per-atom grain-ID (if `--final-with-grain`)
