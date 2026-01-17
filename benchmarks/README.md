# EXPoly Benchmarks

This directory contains benchmarking scripts to measure EXPoly performance on different data sizes.

## Quick Start

```bash
# Run benchmarks with default settings
python benchmarks/benchmark.py

# Generate test data first
python benchmarks/generate_toy_data.py --sizes 20 50 100
```

## Benchmark Scripts

- **`benchmark.py`**: Main benchmarking script that measures conversion time
- **`generate_toy_data.py`**: Generate synthetic Dream3D HDF5 files of different sizes

## Output

Benchmarks generate:
- `benchmark_results.csv`: CSV file with timing results
- `benchmark_results.json`: JSON file with detailed results

## Benchmark Metrics

The benchmark measures:
- **Frame loading time**: Time to load and parse HDF5 file
- **Carve time**: Time to generate atomistic lattices for all grains
- **Polish time**: Time to process and generate final LAMMPS data
- **Total time**: End-to-end pipeline time
- **Output size**: Number of atoms in final.data

## Example Results

See `docs/benchmarks.md` for example benchmark results and performance characteristics.
