#!/usr/bin/env python3
"""
Benchmark EXPoly conversion performance on different data sizes.

Measures:
- Frame loading time
- Carve time
- Polish time
- Total time
- Output atom count
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from expoly.cli import _build_frame_for_carve, _carve_all, _pick_grain_ids, _parse_range
from expoly.frames import Frame
from expoly.polish import PolishConfig, polish_pipeline


def benchmark_single_file(
    dream3d_path: Path,
    hx: tuple[int, int],
    hy: tuple[int, int],
    hz: tuple[int, int],
    lattice: str = "FCC",
    ratio: float = 1.5,
    lattice_constant: float = 3.524,
    workers: int = 1,
    seed: int | None = None,
) -> Dict[str, float | int]:
    """
    Benchmark a single Dream3D file conversion.
    
    Returns a dictionary with timing and output metrics.
    """
    import tempfile
    
    results: Dict[str, float | int] = {}
    
    # 1. Frame loading
    t0 = time.perf_counter()
    try:
        frame = _build_frame_for_carve(str(dream3d_path), None, None)
        t_frame = time.perf_counter() - t0
        results['frame_load_time'] = t_frame
        
        # Get volume info
        results['volume_size'] = frame.HX_lim * frame.HY_lim * frame.HZ_lim
        results['hx_lim'] = frame.HX_lim
        results['hy_lim'] = frame.HY_lim
        results['hz_lim'] = frame.HZ_lim
    except Exception as e:
        results['error'] = f"Frame loading failed: {e}"
        return results
    
    # 2. Carve
    t0 = time.perf_counter()
    try:
        gids = _pick_grain_ids(frame, hx, hy, hz)
        results['n_grains'] = len(gids)
        
        df_all = _carve_all(
            dream3d=Path(dream3d_path),
            hx=hx, hy=hy, hz=hz,
            lattice=lattice,
            ratio=ratio,
            extend=False,
            unit_extend_ratio=3,
            workers=workers,
            seed=seed,
            voxel_csv=None,
            h5_grain_dset=None,
        )
        t_carve = time.perf_counter() - t0
        results['carve_time'] = t_carve
        results['carved_atoms'] = len(df_all)
        
        # Save to temp CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_csv = Path(f.name)
            df_all.to_csv(temp_csv, header=False, index=False)
        
    except Exception as e:
        results['error'] = f"Carve failed: {e}"
        return results
    
    # 3. Polish
    t0 = time.perf_counter()
    try:
        scan_ratio = lattice_constant / ratio
        pcfg = PolishConfig(
            scan_ratio=scan_ratio,
            cube_ratio=ratio,
            hx_range=hx,
            hy_range=hy,
            hz_range=hz,
            real_extent=False,
            unit_extend_ratio=3,
            ovito_cutoff=1.6,
            atom_mass=58.6934,
            keep_tmp=False,
            overwrite=True,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            paths = {
                "tmp_in": tmpdir_path / "tmp_polish.in.data",
                "ovito_mask": tmpdir_path / "overlap_mask.txt",
                "ovito_psc": tmpdir_path / "ovito_cleaned.data",
                "final_lmp": tmpdir_path / "final.data",
            }
            
            final_path = polish_pipeline(
                temp_csv,
                pcfg,
                paths,
                final_with_grain=False,
            )
            
            # Count atoms in final file
            if final_path.exists():
                with open(final_path, 'r') as f:
                    for line in f:
                        if 'atoms' in line.lower() and not line.startswith('#'):
                            try:
                                n_atoms = int(line.split()[0])
                                results['final_atoms'] = n_atoms
                                break
                            except (ValueError, IndexError):
                                pass
        
        t_polish = time.perf_counter() - t0
        results['polish_time'] = t_polish
        
    except Exception as e:
        results['error'] = f"Polish failed: {e}"
        return results
    finally:
        # Cleanup
        if 'temp_csv' in locals() and temp_csv.exists():
            temp_csv.unlink()
    
    # Total time
    results['total_time'] = results.get('frame_load_time', 0) + \
                           results.get('carve_time', 0) + \
                           results.get('polish_time', 0)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark EXPoly conversion performance'
    )
    parser.add_argument(
        '--dream3d',
        type=Path,
        required=True,
        help='Dream3D HDF5 file to benchmark'
    )
    parser.add_argument(
        '--hx',
        type=str,
        default='0:20',
        help='HX range (default: 0:20)'
    )
    parser.add_argument(
        '--hy',
        type=str,
        default='0:20',
        help='HY range (default: 0:20)'
    )
    parser.add_argument(
        '--hz',
        type=str,
        default='0:20',
        help='HZ range (default: 0:20)'
    )
    parser.add_argument(
        '--lattice',
        choices=['FCC', 'BCC', 'DIA'],
        default='FCC',
        help='Lattice type (default: FCC)'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.5,
        help='Lattice ratio (default: 1.5)'
    )
    parser.add_argument(
        '--lattice-constant',
        type=float,
        default=3.524,
        help='Lattice constant in Å (default: 3.524)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers (default: 1)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('benchmark_results.csv'),
        help='Output CSV file (default: benchmark_results.csv)'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=None,
        help='Also output JSON file'
    )
    
    args = parser.parse_args()
    
    # Parse ranges
    hx = _parse_range(args.hx)
    hy = _parse_range(args.hy)
    hz = _parse_range(args.hz)
    
    print(f"Benchmarking: {args.dream3d}")
    print(f"  H ranges: HX={hx}, HY={hy}, HZ={hz}")
    print(f"  Lattice: {args.lattice}, ratio={args.ratio}")
    print()
    
    results = benchmark_single_file(
        dream3d_path=args.dream3d,
        hx=hx,
        hy=hy,
        hz=hz,
        lattice=args.lattice,
        ratio=args.ratio,
        lattice_constant=args.lattice_constant,
        workers=args.workers,
    )
    
    # Print results
    print("=" * 60)
    print("Benchmark Results")
    print("=" * 60)
    
    if 'error' in results:
        print(f"✗ Error: {results['error']}")
        return 1
    
    print(f"Volume size: {results.get('hx_lim', 'N/A')} × "
          f"{results.get('hy_lim', 'N/A')} × {results.get('hz_lim', 'N/A')}")
    print(f"Number of grains: {results.get('n_grains', 'N/A')}")
    print(f"\nTiming (seconds):")
    print(f"  Frame loading: {results.get('frame_load_time', 0):.3f}")
    print(f"  Carve:         {results.get('carve_time', 0):.3f}")
    print(f"  Polish:        {results.get('polish_time', 0):.3f}")
    print(f"  Total:         {results.get('total_time', 0):.3f}")
    print(f"\nOutput:")
    print(f"  Carved atoms:  {results.get('carved_atoms', 'N/A'):,}")
    print(f"  Final atoms:   {results.get('final_atoms', 'N/A'):,}")
    
    # Save to CSV
    df = pd.DataFrame([results])
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")
    
    # Save to JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to: {args.json}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
