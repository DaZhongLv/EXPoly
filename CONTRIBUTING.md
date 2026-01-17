# Contributing to EXPoly

Thank you for your interest in contributing to EXPoly!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DaZhongLv/EXPoly.git
   cd EXPoly
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Run `ruff check .` before committing
- Keep functions focused and well-documented

## Testing

- Add tests for new features in `tests/`
- Run tests with: `pytest tests/ -v`
- Ensure all tests pass before submitting a PR

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Add/update tests as needed
4. Update documentation if necessary
5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- EXPoly version
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error messages or logs

## Questions?

Feel free to open an issue for questions or discussions.
