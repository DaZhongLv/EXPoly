"""High-level programmatic API for EXPoly pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def run(
    dream3d: str | Path,
    hx: tuple[int, int],
    hy: tuple[int, int],
    hz: tuple[int, int],
    lattice_constant: float,
    *,
    lattice: str = "FCC",
    ratio: float = 1.5,
    workers: Optional[int] = None,
    seed: Optional[int] = None,
    extend: bool = False,
    unit_extend_ratio: int = 3,
    real_extent: bool = False,
    ovito_cutoff: float = 1.6,
    atom_mass: float = 58.6934,
    keep_tmp: bool = False,
    final_with_grain: bool = False,
    outdir: Optional[str | Path] = None,
    voxel_csv: Optional[str | Path] = None,
    h5_grain_dset: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """
    Run the EXPoly carve + polish pipeline programmatically.
    
    This is a high-level API that wraps the CLI functionality.
    
    Parameters
    ----------
    dream3d : str | Path
        Path to Dream3D HDF5 file
    hx, hy, hz : tuple[int, int]
        H-space crop ranges (inclusive), e.g., (0, 50)
    lattice_constant : float
        Physical lattice constant in Å (e.g., 3.524 for Ni)
    lattice : str, default="FCC"
        Lattice type: "FCC", "BCC", or "DIA"
    ratio : float, default=1.5
        Lattice-to-voxel scale ratio
    workers : int, optional
        Parallel workers for carving (default: CPU count)
    seed : int, optional
        Random seed for reproducible carving
    extend : bool, default=False
        Use extended-neighborhood pipeline
    unit_extend_ratio : int, default=3
        Unit extend ratio (recommend odd numbers)
    real_extent : bool, default=False
        Multiply H ranges by unit-extend-ratio in polish
    ovito_cutoff : float, default=1.6
        OVITO overlap cutoff distance in Å
    atom_mass : float, default=58.6934
        Atom mass for LAMMPS (default: Ni)
    keep_tmp : bool, default=False
        Keep temporary files
    final_with_grain : bool, default=False
        Write additional final.dump with grain-ID
    outdir : str | Path, optional
        Output directory (default: runs/expoly-<timestamp>)
    voxel_csv : str | Path, optional
        Optional voxel grid CSV
    h5_grain_dset : str, optional
        Custom grain-ID dataset name (default: FeatureIds)
    verbose : bool, default=False
        Enable verbose logging
    
    Returns
    -------
    Path
        Path to the generated final.data file
    
    Examples
    --------
    >>> from expoly import run
    >>> final_path = run(
    ...     dream3d="data.dream3d",
    ...     hx=(0, 50), hy=(0, 50), hz=(0, 50),
    ...     lattice_constant=3.524,
    ...     lattice="FCC",
    ...     ratio=1.5,
    ... )
    >>> print(f"Output: {final_path}")
    """
    import os
    from expoly.cli import run_noninteractive
    from argparse import Namespace
    
    # Convert to Namespace for compatibility with run_noninteractive
    ns = Namespace(
        dream3d=Path(dream3d),
        voxel_csv=Path(voxel_csv) if voxel_csv else None,
        h5_grain_dset=h5_grain_dset,
        hx=hx,
        hy=hy,
        hz=hz,
        lattice=lattice,
        ratio=ratio,
        lattice_constant=lattice_constant,
        workers=workers if workers is not None else (os.cpu_count() or 1),
        seed=seed,
        extend=extend,
        unit_extend_ratio=unit_extend_ratio,
        real_extent=real_extent,
        ovito_cutoff=ovito_cutoff,
        atom_mass=atom_mass,
        keep_tmp=keep_tmp,
        outdir=Path(outdir) if outdir else None,
        final_with_grain=final_with_grain,
        verbose=verbose,
    )
    
    # Run the pipeline
    result = run_noninteractive(ns)
    if result != 0:
        raise RuntimeError(f"Pipeline failed with exit code {result}")
    
    # Find the output directory (created by run_noninteractive)
    import time
    if outdir:
        base = Path(outdir)
    else:
        base = Path("runs")
    
    # Find the most recent run directory
    run_dirs = sorted(base.glob("expoly-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise RuntimeError("Could not find output directory")
    
    final_path = run_dirs[0] / "final.data"
    if not final_path.exists():
        raise RuntimeError(f"Output file not found: {final_path}")
    
    return final_path
