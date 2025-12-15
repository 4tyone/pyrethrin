# Contributing to Pyrethrin

Thank you for your interest in contributing to Pyrethrin!

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/pyrethrin.git
cd pyrethrin

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyrethrin

# Run specific test file
pytest tests/test_decorators.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## Project Structure

```
pyrethrin/
├── pyrethrin/
│   ├── __init__.py       # Public API exports
│   ├── __main__.py       # CLI entry point
│   ├── _ast_dump.py      # AST extraction for static analysis
│   ├── async_support.py  # @async_raises and async_match
│   ├── decorators.py     # @raises and @returns_option decorators
│   ├── exceptions.py     # Custom exceptions
│   ├── match.py          # match() function
│   ├── option.py         # Some/Nothing types
│   ├── result.py         # Ok/Err types
│   ├── testing.py        # Test utilities
│   └── bin/              # Bundled pyrethrum binaries
├── tests/
│   ├── test_decorators.py
│   ├── test_combined_decorators.py  # @raises + @returns_option tests
│   ├── test_match.py
│   ├── test_result.py
│   ├── test_async.py
│   ├── test_ast_dump.py
│   └── test_edge_cases.py
└── examples/
    ├── combined_decorators_correct.py         # Proper combined usage
    ├── combined_decorators_missing_handlers.py # Common mistakes
    ├── option_example.py
    └── user_service.py
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow existing code style
- Add tests for new functionality
- Update documentation if needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run linting
ruff check .

# Run type checking
mypy pyrethrin
```

### 4. Submit a Pull Request

- Write a clear PR description
- Reference any related issues
- Ensure all tests pass

## Code Guidelines

### Style

- Follow PEP 8
- Use type hints for all public APIs
- Keep functions focused and small
- Write docstrings for public functions

### Testing

- Write tests for all new functionality
- Use descriptive test names
- Test edge cases
- Aim for high coverage on core logic

### Commits

- Use clear, descriptive commit messages
- Reference issues when relevant (e.g., "Fix #123")
- Keep commits focused on single changes

## Areas for Contribution

- Bug fixes
- Documentation improvements
- Performance optimizations
- Additional test cases
- IDE integrations

## Questions?

Open an issue for any questions or discussions.
