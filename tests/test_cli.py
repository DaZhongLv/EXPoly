"""Tests for expoly.cli module."""

from __future__ import annotations

from pathlib import Path

import pytest

from expoly.cli import _parse_lattice_constant, _parse_range, build_parser


def test_parse_lattice_constant():
    """Test lattice constant parsing."""
    assert _parse_lattice_constant("3.524") == {"FCC": 3.524}
    assert _parse_lattice_constant("FCC:3.524") == {"FCC": 3.524}
    assert _parse_lattice_constant("FCC:3.524,BCC:2.87") == {"FCC": 3.524, "BCC": 2.87}
    assert _parse_lattice_constant("fcc:3.524,bcc:2.87") == {"FCC": 3.524, "BCC": 2.87}


def test_validate_lattice_constants_missing_phase(tmp_dir):
    """Test that validation raises when lattice constant is missing for a phase."""
    import h5py
    import numpy as np

    from expoly.cli import _carve_all, _validate_lattice_constants
    from expoly.frames import Frame

    h5_path = tmp_dir / "phase_test.dream3d"
    z_size, y_size, x_size = 20, 20, 20

    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    feature_ids[0:10, 0:10, 0:10, 0] = 1
    feature_ids[10:20, 10:20, 10:20, 0] = 2

    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    num_neighbors = np.array([1, 1], dtype=np.int32)
    neighbor_list = np.array([2, 1], dtype=np.int32)
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    phases = np.array([0, 1, 2], dtype=np.int32)
    phase_names = np.array(["Unknown", "fcc", "bcc"], dtype="S32")

    with h5py.File(h5_path, "w") as f:
        c = f.create_group("DataContainers")
        v = c.create_group("SyntheticVolumeDataContainer")
        cd = v.create_group("CellData")
        cd.create_dataset("FeatureIds", data=feature_ids)
        cd.create_dataset("EulerAngles", data=euler_angles)
        fd = v.create_group("CellFeatureData")
        fd.create_dataset("NumNeighbors", data=num_neighbors)
        fd.create_dataset("NeighborList", data=neighbor_list)
        fd.create_dataset("Phases", data=phases)
        s = c.create_group("StatsGeneratorDataContainer")
        e = s.create_group("CellEnsembleData")
        e.create_dataset("PhaseName", data=phase_names)
        g = v.create_group("_SIMPL_GEOMETRY")
        g.create_dataset("DIMENSIONS", data=dimensions)

    frame = Frame(str(h5_path))
    gids = np.array([1, 2])

    # Only FCC provided, but grain 2 is BCC -> should raise
    with pytest.raises(RuntimeError, match="Lattice constant not specified"):
        _validate_lattice_constants(frame, gids, {"FCC": 3.524}, "FCC")

    # Both provided -> should not raise
    _validate_lattice_constants(frame, gids, {"FCC": 3.524, "BCC": 2.87}, "FCC")


def test_carve_all_dual_phase(tmp_dir):
    """Test _carve_all with dual-phase (FCC+BCC) Dream3D data."""
    import h5py
    import numpy as np

    from expoly.cli import _carve_all

    h5_path = tmp_dir / "dual_phase.dream3d"
    z_size, y_size, x_size = 20, 20, 20

    feature_ids = np.zeros((z_size, y_size, x_size, 1), dtype=np.int32)
    feature_ids[0:10, 0:10, 0:10, 0] = 1
    feature_ids[10:20, 10:20, 10:20, 0] = 2

    euler_angles = np.zeros((z_size, y_size, x_size, 3), dtype=np.float32)
    euler_angles[0:10, 0:10, 0:10, :] = [0.0, 0.0, 0.0]
    euler_angles[10:20, 10:20, 10:20, :] = [np.pi / 4, np.pi / 6, np.pi / 3]

    num_neighbors = np.array([1, 1], dtype=np.int32)
    neighbor_list = np.array([2, 1], dtype=np.int32)
    dimensions = np.array([x_size, y_size, z_size], dtype=np.int32)
    phases = np.array([0, 1, 2], dtype=np.int32)
    phase_names = np.array(["Unknown", "fcc", "bcc"], dtype="S32")

    with h5py.File(h5_path, "w") as f:
        c = f.create_group("DataContainers")
        v = c.create_group("SyntheticVolumeDataContainer")
        cd = v.create_group("CellData")
        cd.create_dataset("FeatureIds", data=feature_ids)
        cd.create_dataset("EulerAngles", data=euler_angles)
        fd = v.create_group("CellFeatureData")
        fd.create_dataset("NumNeighbors", data=num_neighbors)
        fd.create_dataset("NeighborList", data=neighbor_list)
        fd.create_dataset("Phases", data=phases)
        s = c.create_group("StatsGeneratorDataContainer")
        e = s.create_group("CellEnsembleData")
        e.create_dataset("PhaseName", data=phase_names)
        g = v.create_group("_SIMPL_GEOMETRY")
        g.create_dataset("DIMENSIONS", data=dimensions)

    df = _carve_all(
        dream3d=h5_path,
        hx=(0, 19),
        hy=(0, 19),
        hz=(0, 19),
        lattice="FCC",
        ratio=1.5,
        extend=False,
        unit_extend_ratio=3,
        workers=1,
        seed=None,
        voxel_csv=None,
        h5_grain_dset=None,
        h5_euler_dset=None,
        lattice_constants={"FCC": 3.524, "BCC": 2.87},
    )

    assert "lattice" in df.columns
    lattices = df["lattice"].unique().tolist()
    assert "FCC" in lattices
    assert "BCC" in lattices
    assert len(df) > 0


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

    args = parser.parse_args(
        [
            "run",
            "--dream3d",
            str(dummy_file),
            "--hx",
            "0:10",
            "--hy",
            "0:10",
            "--hz",
            "0:10",
            "--lattice-constant",
            "3.524",
        ]
    )

    assert args.command == "run"
    assert args.dream3d == dummy_file
    assert args.hx == (0, 10)
    assert args.hy == (0, 10)
    assert args.hz == (0, 10)
    assert args.lattice_constant == {"FCC": 3.524}
    assert args.lattice == "FCC"  # Default
    assert args.ratio == 1.5  # Default


def test_cli_run_multi_phase_args(tmp_dir: Path):
    """Test run command parses multi-phase lattice-constant format."""
    parser = build_parser()
    dummy_file = tmp_dir / "test.dream3d"
    dummy_file.touch()

    args = parser.parse_args(
        [
            "run",
            "--dream3d",
            str(dummy_file),
            "--hx",
            "0:10",
            "--hy",
            "0:10",
            "--hz",
            "0:10",
            "--lattice-constant",
            "FCC:3.524,BCC:2.87",
        ]
    )

    assert args.command == "run"
    assert args.lattice_constant == {"FCC": 3.524, "BCC": 2.87}


def test_cli_defaults(tmp_dir: Path):
    """Test CLI default values."""
    parser = build_parser()
    dummy_file = tmp_dir / "test.dream3d"
    dummy_file.touch()

    args = parser.parse_args(
        [
            "run",
            "--dream3d",
            str(dummy_file),
            "--hx",
            "0:10",
            "--hy",
            "0:10",
            "--hz",
            "0:10",
            "--lattice-constant",
            "3.524",
        ]
    )

    assert args.lattice == "FCC"
    assert args.ratio == 1.5
    assert args.ovito_cutoff == 1.6
    assert args.atom_mass == 58.6934
    assert args.keep_tmp is False
    assert args.final_with_grain is False
    assert args.verbose is False
