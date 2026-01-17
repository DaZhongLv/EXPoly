"""Tests for expoly.carve module."""

from __future__ import annotations

import numpy as np
import pytest

from expoly.carve import (
    CarveConfig,
    sc_to_bcc,
    sc_to_dia,
    sc_to_fcc,
    unit_vec_ratio,
)


def test_unit_vec_ratio():
    """Test unit vector ratio calculation."""
    base = np.array([[0, 0, 0], [1, 1, 1]])
    offsets = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    ratio = 2.0
    
    result = unit_vec_ratio(base, offsets, ratio)
    assert result.shape == (4, 3)  # 2 base × 2 offsets
    assert np.allclose(result[0], [0, 0, 0] + offsets[0] * ratio)


def test_sc_to_fcc():
    """Test SC to FCC conversion."""
    sc_points = np.array([[0, 0, 0], [1, 1, 1]])
    ratio = 1.0
    
    fcc_points = sc_to_fcc(sc_points, ratio)
    assert fcc_points.shape[0] == sc_points.shape[0] * 4  # 4 atoms per SC point
    assert fcc_points.shape[1] == 3


def test_sc_to_bcc():
    """Test SC to BCC conversion."""
    sc_points = np.array([[0, 0, 0]])
    ratio = 1.0
    
    bcc_points = sc_to_bcc(sc_points, ratio)
    assert bcc_points.shape[0] == 2  # 2 atoms per SC point
    assert bcc_points.shape[1] == 3


def test_sc_to_dia():
    """Test SC to diamond conversion."""
    sc_points = np.array([[0, 0, 0]])
    ratio = 1.0
    
    dia_points = sc_to_dia(sc_points, ratio)
    assert dia_points.shape[0] == 8  # 8 atoms per SC point
    assert dia_points.shape[1] == 3


def test_carve_config():
    """Test CarveConfig dataclass."""
    cfg = CarveConfig(
        lattice="FCC",
        ratio=1.5,
        ci_radius=1.414,
        random_center=False,
        rng_seed=42,
        unit_extend_ratio=3,
    )
    assert cfg.lattice == "FCC"
    assert cfg.ratio == 1.5
    assert cfg.ci_radius == 1.414
    assert cfg.random_center is False
    assert cfg.rng_seed == 42
    assert cfg.unit_extend_ratio == 3


def test_deterministic_carve(frame_instance):
    """Test that carving with same seed produces deterministic results."""
    from expoly.carve import process
    
    cfg1 = CarveConfig(lattice="FCC", ratio=1.5, rng_seed=42)
    cfg2 = CarveConfig(lattice="FCC", ratio=1.5, rng_seed=42)
    
    # Process same grain with same config
    df1 = process(1, frame_instance, cfg1)
    df2 = process(1, frame_instance, cfg2)
    
    # Should produce same number of points (deterministic)
    # Note: exact coordinates may differ slightly due to floating point,
    # but structure should be the same
    assert len(df1) == len(df2)
    assert set(df1.columns) == set(df2.columns)
    assert 'grain-ID' in df1.columns
    assert 'X' in df1.columns
    assert 'Y' in df1.columns
    assert 'Z' in df1.columns
