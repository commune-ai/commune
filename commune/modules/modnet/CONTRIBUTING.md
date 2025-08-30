# Contributing to Mod-Net

Thank you for your interest in contributing to Mod-Net! We welcome all contributions, whether they're bug reports, feature requests, documentation improvements, or code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone --recursive https://github.com/your-username/mod-net-modules.git
   cd mod-net-modules/modules
   ```
3. **Set up** the development environment (see [Development Guide](./docs/DEVELOPMENT.md))
4. **Create a branch** for your changes
   ```bash
   git checkout -b feature/amazing-feature
   ```

## Reporting Issues

Before creating an issue, please check if a similar issue already exists.

### Bug Reports

When reporting a bug, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Rust/Python versions, etc.)
6. Any relevant logs or error messages

### Security Issues

Please report security issues to security@mod-net.io. We'll address them as soon as possible.

## Feature Requests

We welcome feature requests! Please include:

1. A clear, descriptive title
2. The problem you're trying to solve
3. A detailed description of the proposed solution
4. Any alternative solutions you've considered
5. Additional context or examples

## Development Workflow

### Branch Naming

Use the following prefixes for branch names:

- `feature/` - New features or enhancements
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or improvements
- `chore/` - Maintenance tasks

Example: `feature/add-login-page`

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code changes that don't add features or fix bugs
- `test`: Adding or modifying tests
- `chore`: Changes to build process or auxiliary tools

Example:
```
feat(auth): add JWT authentication

- Implement JWT token generation
- Add authentication middleware
- Update API documentation

Resolves: #123
```

## Code Style

### Rust

1. Run the formatter:
   ```bash
   cargo fmt --all
   ```

2. Run the linter:
   ```bash
   cargo clippy --all-targets -- -D warnings
   ```

### Python

1. Format code with Black and isort:
   ```bash
   black .
   isort . --profile black
   ```

2. Run linters:
   ```bash
   ruff check .
   mypy .
   ```

## Testing

### Writing Tests

- Write unit tests for new functionality
- Include integration tests for critical paths
- Follow the existing test patterns
- Test edge cases and error conditions

### Running Tests

#### Rust Tests
```bash
# Run all tests
cargo test --all

# Run specific test
cargo test -p pallet-module-registry
```

#### Python Tests
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_module_registry.py -v
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Ensure your code follows the style guide
4. Open a pull request against the `main` branch
5. Request reviews from maintainers
6. Address any feedback
7. Once approved, a maintainer will merge your PR

### PR Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All CI checks pass
- [ ] Changes have been tested

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: Backwards-compatible functionality
- PATCH: Backwards-compatible bug fixes

### Creating a Release

1. Create a release branch:
   ```bash
   git checkout -b release/v1.0.0
   ```

2. Update version in:
   - `Cargo.toml`
   - `commune-ipfs/pyproject.toml`
   - `CHANGELOG.md`

3. Create a pull request with the version update

4. After merging, create a GitHub release with:
   - Version tag (v1.0.0)
   - Release notes
   - Binary attachments (if applicable)

## Community

### Getting Help

- Join our [Discord server](https://discord.gg/modnet)
- Check the [FAQ](./docs/FAQ.md)
- Search existing issues

### Becoming a Maintainer

We're always looking for dedicated contributors to become maintainers. If you're interested:

1. Make several high-quality contributions
2. Help review pull requests
3. Participate in discussions
4. Ask to be added as a maintainer

## License

By contributing, you agree that your contributions will be licensed under the [MIT-0 License](LICENSE).
