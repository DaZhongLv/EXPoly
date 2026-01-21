"""Tests for expoly.cli module."""

from __future__ import annotations

from pathlib import Path

import pytest

from expoly.cli import _parse_range, build_parser


def test_parse_range():
    """Test range parsing."""
    assert _parse_range("0:50") == (0, 50)
    assert _parse_range("10:100") == (10, 100)
    assert _parse_range("[0:50]") == (0, 50)  # Tolerate brackets
    assert _parse_range(" 0 : 50 ") == (0, 50)  # Tolerate spaces


def test_parse_range_invalid():
    """Test invalid range parsing."""
    with pytest.raises(Exception):  # argparse.ArgumentTypeError
        _parse_range("invalid")
    with pytest.raises(Exception):
        _parse_range("50")  # Missing colon


def test_build_parser():
    """Test argument parser construction."""
    parser = build_parser()
    assert parser is not None
    assert parser.prog == "expoly"


def test_cli_help():
    """Test that --help works."""
    parser = build_parser()

    # Should not raise
    try:
        parser.parse_args(["--help"])
    except SystemExit:
        pass  # argparse calls sys.exit on --help


def test_cli_run_required_args():
    """Test that run command requires essential arguments."""
    parser = build_parser()

    # Missing required args should raise
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--dream3d", "test.dream3d"])
        # Missing --hx, --hy, --hz, --lattice-constant


def test_cli_run_minimal_args(tmp_dir: Path):
    """Test run command with minimal required arguments."""
    parser = build_parser()

    # Create dummy file
    dummy_file = tmp_dir / "test.dream3d"
    dummy_file.touch()

    args = parser.parse_args([
        "run",
        "--dream3d", str(dummy_file),
        "--hx", "0:10",
        "--hy", "0:10",
        "--hz", "0:10",
        "--lattice-constant", "3.524",
    ])

    assert args.command == "run"
    assert args.dream3d == dummy_file
    assert args.hx == (0, 10)
    assert args.hy == (0, 10)
    assert args.hz == (0, 10)
    assert args.lattice_constant == 3.524
    assert args.lattice == "FCC"  # Default
    assert args.ratio == 1.5  # Default


def test_cli_defaults(tmp_dir: Path):
    """Test CLI default values."""
    parser = build_parser()
    dummy_file = tmp_dir / "test.dream3d"
    dummy_file.touch()

    args = parser.parse_args([
        "run",
        "--dream3d", str(dummy_file),
        "--hx", "0:10",
        "--hy", "0:10",
        "--hz", "0:10",
        "--lattice-constant", "3.524",
    ])

    assert args.lattice == "FCC"
    assert args.ratio == 1.5
    assert args.ovito_cutoff == 1.6
    assert args.atom_mass == 58.6934
    assert args.keep_tmp is False
    assert args.final_with_grain is False
    assert args.verbose is False
