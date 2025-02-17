# Contributing to Commune

We're excited that you're interested in contributing to Commune! This guide will help you set up your development environment and understand our contribution workflow.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Development Guidelines](#development-guidelines)
- [Code Quality Standards](#code-quality-standards)
- [Contribution Workflow](#contribution-workflow)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Development Environment Setup

### Prerequisites

- Python (3.9 <= version <= 3.12)
- [Poetry](https://python-poetry.org/docs/#installation) (recommended) or pip
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/commune.git
   cd commune
   ```

2. Set up your development environment:

   Using Poetry (recommended):
   ```bash
   poetry install --with dev
   ```

   Using pip:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Guidelines

### Code Style

We maintain strict code quality standards to ensure maintainability and consistency:

1. Code Formatting:
   ```bash
   # Format code using Black
   poetry run black commune

   # Sort imports
   poetry run isort commune
   ```

2. Type Checking:
   ```bash
   # Run type checker
   poetry run mypy commune
   ```

3. Linting:
   ```bash
   # Run linter
   poetry run flake8 commune
   ```

## Code Quality Standards

### Type Hints

All new code should include type hints:

```python
from typing import List, Optional

def process_data(data: List[str], config: Optional[dict] = None) -> bool:
    return True
```

### Documentation

- All modules, classes, and functions should have docstrings
- Follow Google style docstring format
- Include usage examples for complex functionality

## Contribution Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them following our commit message convention:
   ```
   <type>(<scope>): <description>

   [optional body]

   [optional footer]
   ```

   Types:
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `style`: Formatting changes
   - `refactor`: Code refactoring
   - `test`: Adding or modifying tests
   - `chore`: Maintenance tasks

   Example:
   ```
   feat(auth): implement JWT authentication

   - Add JWT token generation
   - Implement token validation
   - Add user authentication middleware

   Closes #123
   ```

3. Push your changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new features
4. Follow the pull request template
5. Link related issues

## Testing Guidelines

1. Write tests for new features:
   ```python
   def test_feature():
       # Arrange
       data = prepare_test_data()
       
       # Act
       result = process_data(data)
       
       # Assert
       assert result is True
   ```

2. Run tests:
   ```bash
   poetry run pytest
   ```

3. Check test coverage:
   ```bash
   poetry run pytest --cov=commune
   ```

## Documentation

- Update documentation for new features
- Include code examples
- Add docstrings to all public APIs
- Update README.md if needed

## Community Guidelines

### Issue Reporting

When reporting issues, please include:

1. Steps to reproduce
2. Expected behavior
3. Actual behavior
4. Environment details
5. Relevant logs or screenshots

### Communication

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow our code of conduct

## License

By contributing to Commune, you agree that your contributions will be licensed under our project's MIT License.

## Questions or Need Help?

- Open a [GitHub Discussion](https://github.com/commune-ai/commune/discussions)
- Join our [Discord Community](https://discord.gg/commune-ai-941362322000203776)

Thank you for contributing to Commune! 🚀