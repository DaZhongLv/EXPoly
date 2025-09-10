# EXPoly

EXPoly reconstructs atomistic lattices from **Dream3D HDF5 voxel data**, merges grains, and exports a **clean LAMMPS data** file using **OVITO** overlap removal.

- **Carve**: build an oriented lattice for each grain (FCC / BCC / diamond cubic).
- **Polish**: crop & scale to physical units, assemble all grains, then remove near-duplicate/too-close atoms at grain boundaries via OVITO; finally write a consistent 'final.data'.

---

## 1) Requirements

- Python **3.10+** (3.11 recommended)
- macOS / Linux / WSL
- Packages (installed via pip):
  - 'numpy', 'pandas', 'scipy', 'h5py', 'jinja2'
  - **'ovito'** (mandatory for polish)
  - optional: 'plotly' (for your own plotting)

---

## 2) Install with a virtual environment

# clone
git clone https://github.com/DaZhongLv/EXPoly.git
cd EXPoly

# create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# install
pip install --upgrade pip
pip install -e .
pip install ovito

---

## 3) Input data (Dream3D HDF5 voxel data)

This tool expects voxel-based experimental data from Dream3D. Minimum datasets:

DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds
3D array (z,y,x,1) of positive grain IDs (0 means void/background).

DataContainers/SyntheticVolumeDataContainer/CellData/CellEulerAngles
-> 3D array (z,y,x,3) of Euler angles (Bunge) per voxel.
-> Unit: the pipeline assumes radians. If your file stores degrees, convert inside the reader (the included reader does this if needed).

The pipeline will compute per-grain average Euler to orient lattices.

---

## 4) Quick start (one-line run)
expoly run \
  --dream3d /abs/path/to/Alpoly_elongate.dream3d \
  --hx 0:50 --hy 0:50 --hz 0:50 \
  --lattice FCC --ratio 1.5 \
  --workers 8 \ # the number of CPU will be used in parallel
  --lattice-constant 3.524 # for Ni lattice

What this does

1.Read Dream3D HDF5 voxel data (--dream3d)
(i.e., voxel-based experimental data with FeatureIds and CellEulerAngles).

2.Crop by H-ranges (--hx/--hy/--hz use start:end, e.g. 0:50)
H-ranges are voxel coordinate intervals in H-space (voxel index space), inclusive.

3.Carve per grain with the choice in three different knids of lattice --lattice (FCC/BCC/DIA) with user defined spacing --ratio.
Currently supported structures are cubic families: FCC, BCC, and diamond cubic.
ratio definition: the ratio between the constructed atomic lattice constant and the unit length of your voxel grid.
-> If one atomistic unit cell exactly “covers” one voxel, ratio = 1.
-> If one atomistic unit cell spans a 2×2×2 block of voxels, ratio = 2.
-> The number of ratio will directly influence the converted grain size, the larger the ratio, the smaller the grain size

Polish & assemble: compute scan_ratio = lattice_constant / ratio then write tmp_polish.in.data (full LAMMPS data), run OVITO to remove overlapping / too-close atoms at grain boundaries, and finally rebuild a clean final.data (correct atom count and box).

-> OVITO neighbor cutoff should be smaller than the nearest-neighbor distance of your chosen lattice to avoid deleting true nearest-neighbor pairs. Nearest-neighbor distances (in units of lattice constant a):
--> FCC: a * √2 / 2 ≈ 0.7071 a
--> BCC: a * √3 / 2 ≈ 0.8660 a
--> Diamond cubic: a * √3 / 4 ≈ 0.4330 a
The default cutoff (1.6) in the tool is safe for Ni with a=3.524 Å (FCC). You can change it if needed.

Outputs：
A new run folder like:
runs/expoly-<timestamp>/
  raw_points.csv         # merged carved points after GB filtering
  tmp_polish.in.data     # pre-OVITO LAMMPS data (you can open it in OVITO)
  ovito_cleaned.data     # OVITO-exported cleaned data
  final.data             # clean LAMMPS data (this is the recommended output)

---

5) CLI flags & defaults
expoly run --dream3d <file>
           --hx a:b --hy c:d --hz e:f
           [--lattice {FCC,BCC,DIA}]   # default: FCC
           [--ratio <float>]           # default: 1.5
           [--lattice-constant <float>]# default: 3.524 (Ni)
           [--workers <int>]           # default: auto (num CPU cores)

--dream3d : Absolute path to .dream3d file.
Alternatively set env var EXPOLY_DREAM3D_PATH=/abs/path/file.dream3d.

--hx/--hy/--hz : crop ranges in H-space (voxel index space), inclusive; format start:end (e.g. 0:50).

--lattice : lattice for carving (FCC default; BCC/DIA diamond cubic supported).

--ratio : lattice-to-voxel scale (aka cube_ratio above).

--lattice-constant : physical lattice constant a (Å). Default 3.524 for Ni (FCC).

--workers : parallel workers for carving (default = CPU cores).

The polish step always runs; its OVITO cutoff has a safe default. If you need to tune it, see src/expoly/polish.py (config option ovito_cutoff).






