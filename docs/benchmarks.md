# EXPoly Benchmarks

This document provides benchmark results and performance characteristics for EXPoly.

## Benchmark Setup

Benchmarks are run using synthetic Dream3D HDF5 files of varying sizes. Each benchmark measures:

- **Frame loading time**: Time to load and parse the HDF5 file
- **Carve time**: Time to generate atomistic lattices for all grains
- **Polish time**: Time to process and generate final LAMMPS data (including OVITO)
- **Total time**: End-to-end pipeline execution time
- **Output metrics**: Number of carved atoms and final atoms

## Running Benchmarks

```bash
# Generate test data
python benchmarks/generate_toy_data.py --sizes 20 50 100

# Run benchmark on a file (using An0new6.dream3d as example)
python benchmarks/benchmark.py \
  --dream3d An0new6.dream3d \
  --hx 0:20 --hy 0:20 --hz 0:20 \
  --lattice FCC --ratio 1.5 \
  --lattice-constant 3.524 \
  --h5-grain-dset FeatureIds \
  --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors \
  --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS
```

## Example Results

*Note: These are example results. Actual performance depends on hardware, data characteristics, and system load.*

### Small Volume (20×20×20 voxels, ~5 grains)

| Metric | Time (seconds) |
|--------|----------------|
| Frame loading | ~0.1 |
| Carve | ~1-2 |
| Polish | ~2-3 |
| **Total** | **~3-5** |

- Output: ~10,000-50,000 atoms (depending on ratio)

### Medium Volume (50×50×50 voxels, ~5 grains)

| Metric | Time (seconds) |
|--------|----------------|
| Frame loading | ~0.2-0.5 |
| Carve | ~5-10 |
| Polish | ~5-10 |
| **Total** | **~10-20** |

- Output: ~100,000-500,000 atoms (depending on ratio)

### Large Volume (100×100×100 voxels, ~5 grains)

| Metric | Time (seconds) |
|--------|----------------|
| Frame loading | ~1-2 |
| Carve | ~30-60 |
| Polish | ~20-40 |
| **Total** | **~50-100** |

- Output: ~1,000,000-5,000,000 atoms (depending on ratio)

## Performance Characteristics

### Scaling Behavior

- **Frame loading**: Scales roughly linearly with volume size (O(n) where n = voxels)
- **Carve**: Scales with:
  - Number of grains (approximately linear)
  - Volume size (approximately linear)
  - Lattice ratio (inverse relationship: larger ratio → fewer atoms → faster)
- **Polish**: Scales with:
  - Number of atoms (approximately O(n log n) due to OVITO neighbor search)
  - OVITO cutoff (larger cutoff → more neighbor checks → slower)

### Memory Usage

- **Frame loading**: ~10-50 MB per million voxels (depending on data types)
- **Carve**: ~100-500 MB per million atoms (depending on intermediate structures)
- **Polish**: ~50-200 MB per million atoms (OVITO processing)

### Optimization Tips

1. **Use appropriate H ranges**: Only process the region you need
2. **Adjust ratio**: Larger ratio → fewer atoms → faster processing
3. **Parallel workers**: Use `--workers` to parallelize carving (memory permitting)
4. **OVITO cutoff**: Use the smallest safe cutoff to reduce polish time

## Factors Affecting Performance

1. **Volume size**: Larger volumes take longer
2. **Number of grains**: More grains → more carving operations
3. **Lattice type**: DIA (8 atoms/unit) > FCC (4 atoms/unit) > BCC (2 atoms/unit) in terms of atoms generated
4. **Ratio**: Larger ratio → fewer atoms → faster
5. **Hardware**: CPU cores, RAM, and disk I/O speed
6. **OVITO**: Polish step is typically the bottleneck for large datasets

## Benchmarking Your Data

To benchmark your own data:

```bash
python benchmarks/benchmark.py \
  --dream3d An0new6.dream3d \
  --hx 0:100 --hy 0:100 --hz 0:100 \
  --lattice FCC \
  --ratio 1.5 \
  --lattice-constant 3.524 \
  --h5-grain-dset FeatureIds \
  --h5-euler-dset EulerAngles \
  --h5-numneighbors-dset NumNeighbors \
  --h5-neighborlist-dset NeighborList2 \
  --h5-dimensions-dset DIMENSIONS \
  --workers 4 \
  --output my_benchmark.csv
```

Results will be saved to CSV for analysis.

## Notes

- Benchmarks are run on a single machine and may vary significantly based on hardware
- OVITO performance depends on the number of atoms and cutoff distance
- Memory usage can be a limiting factor for very large volumes
- For production use, consider processing in chunks or using larger H ranges incrementally
