# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-16

### Added
- Professional repository structure with documentation and examples
- Comprehensive README with quickstart, failure modes, and output descriptions
- Test suite with pytest
- GitHub Actions CI workflow
- `expoly doctor` command for input validation
- Improved error messages with actionable remediation steps
- Examples directory with minimal runnable examples
- Documentation in `docs/` directory
- Benchmarking infrastructure

### Changed
- Version bumped from 0.2.0 to 1.0.0 (major refactoring release)
- Removed unused dependencies (scikit-learn, jinja2)
- Improved CLI help text with grouped options
- Enhanced error handling and diagnostics

### Fixed
- Improved error messages for missing HDF5 datasets
- Better validation of input parameters

## [0.2.0] - Previous Release

### Features
- Initial release with carve and polish pipeline
- Support for FCC, BCC, and diamond cubic lattices
- OVITO integration for overlap removal
- VoxelCSVFrame support for custom voxel grids
- Extended neighborhood pipeline option
