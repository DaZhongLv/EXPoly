"""Tests for expoly.polish module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from expoly.polish import PolishConfig, _load_raw_points, write_lammps_input_data


def test_polish_config():
    """Test PolishConfig dataclass."""
    cfg = PolishConfig(
        scan_ratio=2.0,
        cube_ratio=1.5,
        hx_range=(0, 10),
        hy_range=(0, 10),
        hz_range=(0, 10),
        real_extent=False,
        unit_extend_ratio=3,
        ovito_cutoff=1.6,
        atom_mass=58.6934,
        keep_tmp=False,
        overwrite=True,
    )
    assert cfg.scan_ratio == 2.0
    assert cfg.cube_ratio == 1.5
    assert cfg.hx_range == (0, 10)
    assert cfg.ovito_cutoff == 1.6


def test_load_raw_points(tmp_dir: Path):
    """Test loading raw points CSV."""
    # Create a minimal raw_points.csv
    csv_path = tmp_dir / "raw_points.csv"
    data = """1.0 2.0 3.0 0 0 0 0 1
4.0 5.0 6.0 1 1 1 0 1
7.0 8.0 9.0 2 2 2 2 2
"""
    csv_path.write_text(data)
    
    df = _load_raw_points(csv_path)
    assert len(df) == 3
    assert list(df.columns) == ['X', 'Y', 'Z', 'HX', 'HY', 'HZ', 'margin-ID', 'grain-ID']
    assert df['X'].iloc[0] == 1.0
    assert df['grain-ID'].iloc[0] == 1


def test_write_lammps_input_data(tmp_dir: Path):
    """Test LAMMPS data file generation."""
    # Create raw points CSV
    csv_path = tmp_dir / "raw_points.csv"
    data = """1.0 2.0 3.0 5 5 5 0 1
4.0 5.0 6.0 6 6 6 0 1
7.0 8.0 9.0 7 7 7 2 2
"""
    csv_path.write_text(data)
    
    cfg = PolishConfig(
        scan_ratio=2.0,
        cube_ratio=1.5,
        hx_range=(5, 10),
        hy_range=(5, 10),
        hz_range=(5, 10),
    )
    
    out_path = tmp_dir / "test.data"
    atom_num, box = write_lammps_input_data(csv_path, cfg, out_path)
    
    assert atom_num == 3
    assert len(box) == 6  # xlo, xhi, ylo, yhi, zlo, zhi
    assert out_path.exists()
    
    # Check file content
    content = out_path.read_text()
    assert f"{atom_num} atoms" in content
    assert "Masses" in content
    assert "Atoms # atomic" in content
    assert "1 58.6934" in content  # Default atom mass


def test_lammps_file_structure(tmp_dir: Path):
    """Test that generated LAMMPS file has correct structure."""
    csv_path = tmp_dir / "raw_points.csv"
    data = """1.0 2.0 3.0 5 5 5 0 1
"""
    csv_path.write_text(data)
    
    cfg = PolishConfig(
        scan_ratio=1.0,
        cube_ratio=1.5,
        hx_range=(5, 10),
        hy_range=(5, 10),
        hz_range=(5, 10),
    )
    
    out_path = tmp_dir / "test.data"
    write_lammps_input_data(csv_path, cfg, out_path)
    
    lines = out_path.read_text().splitlines()
    
    # Check header structure
    assert any("atoms" in line.lower() for line in lines)
    assert any("atom types" in line.lower() for line in lines)
    # Check for box bounds (xlo xhi, ylo yhi, zlo zhi)
    all_text = " ".join(lines).lower()
    assert ("xlo xhi" in all_text or "xlo" in all_text)
    assert any("Masses" in line for line in lines)
    assert any("Atoms" in line for line in lines)
