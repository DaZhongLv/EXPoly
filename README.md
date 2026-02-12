# EXPoly

**EXPoly converts experimental microstructure voxel data (Dream3D HDF5) into MD-ready atomistic structures (LAMMPS data files).**

EXPoly reconstructs atomistic lattices from **Dream3D HDF5 voxel data**, merges grains, and exports a **clean LAMMPS data** file using **OVITO** overlap removal.

- **Carve**: build an oriented lattice for each grain (FCC / BCC / diamond cubic).
- **Polish**: crop & scale to physical units, assemble all grains, then remove near-duplicate/too-close atoms at grain boundaries via OVITO; finally write a consistent 'final.data'.

---

## Quickstart (30 seconds)

```bash
# Install
pip install -e .

# Run (using An0new6.dream3d as example)
expoly run \
  --dream3d An0new6.dream3d \
  --hx 100:180 --hy 100:180 --hz 10:60 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524 \
  --workers 2 \
  --h5-grain-dset FeatureIds \
  --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors \
  --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS
```

Output: `runs/expoly-<timestamp>/final.data` (ready for LAMMPS)

---

## Pipeline Overview

```
┌─────────────────┐
│ Dream3D HDF5     │  Input: voxel grid with grain IDs and Euler angles
│ (voxel data)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Carve         │  For each grain:
│   - Read HDF5   │  1. Compute average Euler angles
│   - Select H    │  2. Build oriented lattice (FCC/BCC/DIA)
│   ranges        │  3. Filter by grain boundaries
│   - Build       │
│   lattices      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Polish        │  1. Crop & scale to physical units
│   - Assemble    │  2. Write pre-OVITO LAMMPS data
│   - OVITO       │  3. Remove overlapping atoms
│   deduplication │  4. Generate clean final.data
│   - Final data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAMMPS data    │  Output: final.data (ready for MD simulation)
│  (final.data)   │
└─────────────────┘
```

---

## Installation

### Requirements

- Python **3.10+** (3.11 recommended)
- macOS / Linux / WSL

### Install with a virtual environment

```bash
# Clone repository
git clone https://github.com/DaZhongLv/EXPoly.git
cd EXPoly

# Create & activate venv
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install --upgrade pip
pip install -e .
pip install ovito
```

### Dependencies

Core dependencies (installed automatically):
- `numpy`, `pandas`, `scipy`, `h5py`
- `ovito` (mandatory for polish step)
- `plotly` (optional, for visualization utilities)

---

## Input Data

This tool expects voxel-based experimental data from Dream3D. Minimum datasets:

- **FeatureIds** (or custom name): 3D/4D array (z,y,x,1) of positive grain IDs (0 = void/background)
  - Default path: `DataContainers/*/CellData/FeatureIds`
  - Example (An0new6.dream3d): `FeatureIds`
  - Customize with `--h5-grain-dset`
- **EulerAngles** (or custom name): 3D array (z,y,x,3) of Euler angles (Bunge convention, radians)
  - Default path: `DataContainers/*/CellData/EulerAngles`
  - Example (An0new6.dream3d): `EulerAngles`
  - Customize with `--h5-euler-dset`
- **NumNeighbors** (or custom name): Number of neighbors per grain
  - Default path: `DataContainers/*/CellFeatureData/NumNeighbors`
  - Example (An0new6.dream3d): `NumNeighbors`
  - Customize with `--h5-numneighbors-dset`
- **NeighborList** (or custom name): Neighbor list array
  - Default path: `DataContainers/*/CellFeatureData/NeighborList`
  - Example (An0new6.dream3d): `NeighborList2` (note: different from default!)
  - Customize with `--h5-neighborlist-dset`
- **DIMENSIONS** (or custom name): Volume dimensions [X, Y, Z]
  - Default path: `DataContainers/*/_SIMPL_GEOMETRY/DIMENSIONS`
  - Example (An0new6.dream3d): `DIMENSIONS`
  - Customize with `--h5-dimensions-dset`

**Note**: Dataset names may vary between Dream3D files. Use the `--h5-*-dset` flags to specify custom names. For example, with An0new6.dream3d, use `--h5-neighborlist-dset NeighborList2`.

The pipeline computes per-grain average Euler angles to orient lattices.

---

## Usage

### Basic Command

```bash
expoly run \
  --dream3d An0new6.dream3d \
  --hx 100:180 --hy 100:180 --hz 10:60 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524 \
  --workers 2 \
  --h5-grain-dset FeatureIds \
  --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors \
  --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS
```

### What This Does

1. **Read Dream3D HDF5** (`--dream3d`): Loads voxel-based grain data with FeatureIds and EulerAngles.

2. **Crop by H-ranges** (`--hx/--hy/--hz`): Selects voxel coordinate intervals in H-space (voxel index space), inclusive. Format: `start:end` (e.g., `0:50`).

3. **Carve per grain** (`--lattice`, `--ratio`):
   - Builds oriented lattice for each grain (FCC/BCC/DIA)
   - `--ratio`: Lattice-to-voxel scale ratio
     - `ratio = 1`: One unit cell per voxel
     - `ratio = 2`: One unit cell spans 2×2×2 voxels
     - Larger ratio → smaller grain size in atomistic representation

4. **Polish & assemble**:
   - Computes `scan_ratio = lattice_constant / ratio`
   - Writes `tmp_polish.in.data` (pre-OVITO LAMMPS data)
   - Runs OVITO to remove overlapping/too-close atoms at grain boundaries
   - Generates clean `final.data` (correct atom count and box)

### CLI Flags & Defaults

```bash
expoly run \
  --dream3d <file>              # Required: Path to Dream3D file
  --hx a:b --hy c:d --hz e:f    # Required: H-space crop ranges
  [--lattice {FCC,BCC,DIA}]     # Default: FCC (FCC=Face-Centered cubic, BCC=Body-Centered cubic, DIA=diamond)
  [--ratio <float>]             # Default: 1.5
  [--lattice-constant <float>]  # Required: Physical lattice constant (Å)
  [--workers <int>]             # Default: 2
  [--ovito-cutoff <float>]      # Default: 1.6 (safe for Ni FCC)
  [--atom-mass <float>]         # Default: 58.6934 (Ni)
  [--keep-tmp]                  # Keep intermediate files
  [--final-with-grain]          # Write additional dump with grain-ID
  [--random-orientation]        # Randomize grain orientations (see below)
  [--seed <int>]                # Random seed for reproducibility
  [--verbose]                   # Verbose logging
```

**Advanced options:**
- `--extend`: Use extended-neighborhood pipeline for carving. When enabled, automatically multiplies HX/HY/HZ ranges by unit-extend-ratio in polish step.
- `--unit-extend-ratio <int>`: Unit extend ratio (default: 3, recommend odd numbers)
- `--random-orientation`: Randomize grain orientations. Shuffles grain IDs and reassigns orientations so each grain ID gets a random orientation from the shuffled list. Use with `--seed` for reproducibility. See [Random orientation](#random-orientation) below.
- `--seed <int>`: Random seed for reproducible carving (affects ball grid randomization and, with `--random-orientation`, orientation shuffling)
- `--voxel-csv <file>`: Optional voxel grid CSV (whitespace-separated)
- `--h5-grain-dset <name>`: Custom grain-ID dataset name (default: FeatureIds)
- `--h5-euler-dset <name>`: Custom Euler angles dataset name (default: EulerAngles)
- `--h5-numneighbors-dset <name>`: Custom NumNeighbors dataset name (default: NumNeighbors)
- `--h5-neighborlist-dset <name>`: Custom NeighborList dataset name (default: NeighborList). Example: NeighborList2
- `--h5-dimensions-dset <name>`: Custom DIMENSIONS dataset name (default: DIMENSIONS)

### Random orientation

With **`--random-orientation`**, EXPoly reassigns orientations to grain IDs using a shuffled mapping:

1. **Build orientation list**: Read all selected grain IDs and their orientations from the HDF5 Euler dataset.
2. **Shuffle grain IDs**: Create a shuffled list of grain IDs (optionally using `--seed` for reproducibility).
3. **Reassign orientations**: For each grain ID, find its position in the shuffled list, then use that position to get the corresponding grain ID from the original list, and assign that grain's orientation.

**Example**: If you have grains [3, 5, 7] and shuffled list is [7, 3, 5]:
- Grain ID 3 → position 1 in shuffled → original[1] = 5 → use grain 5's orientation
- Grain ID 5 → position 2 in shuffled → original[2] = 7 → use grain 7's orientation  
- Grain ID 7 → position 0 in shuffled → original[0] = 3 → use grain 3's orientation

This breaks the original grain ID ↔ orientation correspondence while keeping the same set of orientations in the volume.

**Use cases**: Sensitivity studies, null models, or when you want to randomize orientation assignments while preserving the orientation distribution.

**Example command**:
```bash
expoly run --dream3d An0new6.dream3d --hx 100:180 --hy 100:180 --hz 10:60 \
  --lattice FCC --ratio 1.5 --lattice-constant 3.524 \
  --workers 2 \
  --h5-grain-dset FeatureIds --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --random-orientation --seed 42
```

### OVITO Cutoff Guidelines

OVITO neighbor cutoff should be **smaller** than the nearest-neighbor distance of your chosen lattice to avoid deleting legitimate nearest-neighbor pairs.

Nearest-neighbor distances (in units of lattice constant a):
- **FCC**: a × √2 / 2 ≈ 0.7071 a
- **BCC**: a × √3 / 2 ≈ 0.8660 a
- **Diamond cubic**: a × √3 / 4 ≈ 0.4330 a

The default cutoff (1.6) is safe for Ni with a=3.524 Å (FCC). Adjust `--ovito-cutoff` if needed.

### Voronoi Topology Extraction

**Purpose:** Convert the atomistic structure from experimental (Dream3D-derived) data into a **Voronoi-like** representation: a regular voxel grid where each voxel is assigned to a grain, with grain boundaries approximated by planar patches. This smooths the original voxel/atom layout and can be used to reconstruct a new atomistic structure with cleaner grain-boundary topology.

**How it works (brief):** The pipeline reads a LAMMPS dump with per-atom grain IDs, crops the edges, then classifies each atom by k-NN into grain-boundary surface (2 grains), triple-line (3 grains), or quadruple-point (4 grains). From quadruple points and triple-line segments it builds a piecewise planar grain-boundary mesh; each grain is voxelized as the set of grid points inside the half-spaces defined by its adjacent GB patches. The result is a regular voxel grid (CSV) where each voxel is assigned to a grain—a Voronoi-like partitioning that can be fed back into `expoly run --voxel-csv` to reconstruct a smoothed atomistic structure.

**Input**

| Input | Description |
|-------|-------------|
| **LAMMPS dump** (`--dump`) | Single-timestep dump with columns: atom id, type, x, y, z, grain-ID. Typically `final.dump` from `expoly run --final-with-grain`. |

**Output**

| Output | Description |
|--------|-------------|
| **Voxel CSV** (`--output`) | Space-separated CSV with columns: `atom-ID`, `grain-ID`, `voxel-X`, `voxel-Y`, `voxel-Z`, `phase`, `CI`. Integer grid coordinates (0-based, increment by 1 after normalizing by `--voxel-size`). Can be used as `--voxel-csv` in a subsequent `expoly run` to rebuild atoms. |

**Standalone `voronoi` command (two-step flow):**

```bash
expoly voronoi \
  --dump runs/expoly-<timestamp>/final.dump \
  --output runs/expoly-<timestamp>/voro_test.csv \
  --voxel-size 2.0
```

Optional: `--cube-ratio 0.015`, `--k 25`, `--min-other-atoms 4`. See `expoly voronoi --help` for all flags.

**One-shot flow (`--generate-voronoi`):** Run the full pipeline (run → voronoi → run with CSV) in one command so you do not need to run `expoly voronoi` or a second `expoly run --voxel-csv` manually:

```bash
expoly run \
  --dream3d An0new6.dream3d \
  --hx 0:119 --hy 0:119 --hz 0:75 \
  --lattice FCC --ratio 2 --lattice-constant 3.524 \
  --h5-grain-dset FeatureIds --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --generate-voronoi --voronoi-voxel-size 2.0
```

Input: Dream3D file + H-ranges and lattice options. Output: same run directory contains `voronoi.csv` and the final atomistic `final.data` from the second pass.

---

## Outputs

EXPoly creates a new run directory: `runs/expoly-<timestamp>/`

### Generated Files

- **`raw_points.csv`**: All carved atoms after grain boundary filtering (before overlap removal)
  - Columns: X, Y, Z, HX, HY, HZ, margin-ID, grain-ID
  - Useful for debugging and inspection

- **`tmp_polish.in.data`**: Pre-OVITO LAMMPS data file
  - Complete LAMMPS data format (header + Masses + Atoms)
  - Can be opened directly in OVITO for visualization
  - Contains all atoms before overlap removal

- **`ovito_cleaned.data`**: OVITO-processed cleaned data
  - Intermediate file after OVITO overlap removal
  - Used internally to generate final.data

- **`final.data`**: **Clean LAMMPS data file (recommended output)**
  - Correct atom count and box bounds
  - Standard LAMMPS format: `Atoms # atomic` style
  - Ready for molecular dynamics simulations
  - This is the file you should use for LAMMPS

- **`final.dump`** (optional): LAMMPS dump format with per-atom grain-ID
  - Only created if `--final-with-grain` is used
  - Useful for post-processing and analysis
  - Format: `ITEM: ATOMS id type x y z grain-ID`

- **`overlap_mask.txt`**: OVITO selection mask (0=keep, 1=delete)
  - Only kept if `--keep-tmp` is used
  - Useful for debugging overlap removal

---

## Failure Modes & Assumptions

### Common Failure Modes

1. **Missing HDF5 datasets**
   - **Error**: `Dataset named 'FeatureIds' not found`
   - **Cause**: HDF5 file structure doesn't match expected paths
   - **Solution**: Use `--h5-grain-dset` to specify custom dataset name, or check HDF5 structure with `h5dump`

2. **Invalid H ranges**
   - **Error**: `No positive grain id found within the provided H ranges`
   - **Cause**: HX/HY/HZ ranges outside volume dimensions or no grains in selected region
   - **Solution**: Check volume dimensions first, adjust ranges accordingly

3. **OVITO not installed**
   - **Error**: `ovito is required. Install it with 'pip install ovito'`
   - **Cause**: OVITO package missing
   - **Solution**: `pip install ovito`

4. **OVITO cutoff too large**
   - **Symptom**: Legitimate nearest-neighbor atoms deleted
   - **Cause**: `--ovito-cutoff` exceeds nearest-neighbor distance
   - **Solution**: Reduce cutoff (see guidelines above)

5. **Memory issues with large volumes**
   - **Symptom**: Out of memory errors during carving
   - **Cause**: Very large H ranges or many grains
   - **Solution**: Reduce H ranges, use `--workers 1` to reduce memory pressure, or process in chunks

6. **Very large HDF5 file + many workers**
   - **Symptom**: Slow runs, swap, or OOM on HPC
   - **Cause**: Each worker loads a full copy of the HDF5 data into memory. With 16 workers and a 2 GB file, that is ~32 GB RAM.
   - **Solution**: Use **`--workers 1`** (or `--workers 2`) when the Dream3D file is very large. Fewer workers = less total memory; each worker still reuses its loaded data for all assigned grains.

7. **Schema mismatch**
   - **Error**: `Unexpected GrainId ndim` or shape mismatches
   - **Cause**: HDF5 array dimensions don't match expected format
   - **Solution**: Check HDF5 structure, use `--h5-grain-dset` if dataset name differs

### Key Assumptions

1. **Crystal structure**: Only cubic families supported (FCC, BCC, DIA)
   - Non-cubic structures (hexagonal, tetragonal, etc.) are not supported

2. **Single orientation per grain**: Each grain has uniform orientation
   - Sub-grain orientation gradients are not modeled

3. **Lattice constant known**: Physical lattice constant must be provided
   - Default (3.524 Å) is for Ni; adjust for your material

4. **Bunge Euler convention**: Euler angles assumed in Bunge convention (radians)
   - If your data uses degrees or different convention, conversion needed

5. **OVITO mandatory**: Polish step requires OVITO
   - Cannot run full pipeline without OVITO installed

6. **Memory scaling**: Processing time and memory scale with:
   - Number of grains
   - Voxel volume (HX × HY × HZ)
   - Lattice density (ratio parameter)

### Limitations

- **Crystal structures**: Only cubic families (FCC/BCC/DIA)
- **Orientation**: Single orientation per grain (no sub-grain variation)
- **File size**: Large volumes may require significant RAM
- **OVITO dependency**: Cannot run polish without OVITO
- **Deterministic carving**: Random seed only affects ball center, not lattice structure

For more details, see [`docs/use_cases.md`](docs/use_cases.md).

---

## Examples

See the [`examples/`](examples/) directory for runnable examples:

- **`minimal_example.py`**: Minimal working example (uses `An0new6.dream3d` if available locally)
- **`toy_data_generator.py`**: Generate synthetic Dream3D HDF5 files for testing

Run examples:
```bash
cd examples
python minimal_example.py
```

### Voronoi Topology Extraction Example

See the [Voronoi Topology Extraction](#voronoi-topology-extraction) section for purpose, inputs/outputs, and how it works. Two ways to run:

**Two-step (manual):** Input: Dream3D + run options → get `final.dump`. Then input: `final.dump` → output: voxel CSV (e.g. `voro_test.csv`).

```bash
# Step 1 — Input: Dream3D. Output: runs/expoly-<timestamp>/final.dump
expoly run \
  --dream3d An0new6.dream3d \
  --hx 0:119 --hy 0:119 --hz 0:75 \
  --lattice FCC --ratio 2 --lattice-constant 3.524 \
  --workers 3 \
  --h5-grain-dset FeatureIds --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --final-with-grain

# Step 2 — Input: final.dump. Output: voro_test.csv (use as --voxel-csv in a later run if desired)
expoly voronoi \
  --dump runs/expoly-<timestamp>/final.dump \
  --output runs/expoly-<timestamp>/voro_test.csv \
  --voxel-size 2.0
```

**One-shot (`--generate-voronoi`):** Input: Dream3D + same run options + `--generate-voronoi`. Output: same run dir with `voronoi.csv` and final `final.data`.

```bash
expoly run \
  --dream3d An0new6.dream3d \
  --hx 0:119 --hy 0:119 --hz 0:75 \
  --lattice FCC --ratio 2 --lattice-constant 3.524 \
  --h5-grain-dset FeatureIds --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --generate-voronoi --voronoi-voxel-size 2.0
```

**Sample data**: Sample Dream3D files are typically large (100+ MB) and are excluded from Git via `.gitignore`. To get sample data:

1. **Download from CMU Grain Boundary Data Archive**: 
   - Visit: [Nickel Grain Boundary Velocity Data](http://mimp.materials.cmu.edu/~gr20/Grain_Boundary_Data_Archive/Ni_velocity/Ni_velocity.html)
   - Download the "Microstructure Data" archive (367 MB, contains 6 Dream3D files)
   - These are real experimental data from Ni polycrystals (Science, 2021)
   - Extract and use any of the Dream3D files with EXPoly

2. **Use your own data**: Provide your Dream3D HDF5 file

3. **Generate toy data**: `python examples/toy_data_generator.py` (creates small test files for development)

---

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and guidelines.

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License - see [`LICENSE`](LICENSE) for details.

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for version history and changes.