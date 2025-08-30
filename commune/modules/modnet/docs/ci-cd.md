# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the Mod-Net module registry project.

## Overview

Our CI/CD pipeline uses GitHub Actions to ensure code quality, run tests, and validate both Rust and Python components. The pipeline consists of three main workflows:

1. Rust CI (`rust.yml`)
2. Python CI (`python.yml`)
3. Integration Tests (`integration.yml`)

## Workflow Details

### 1. Rust CI Workflow

**File**: `.github/workflows/rust.yml`

**Triggers**:
- Push to main branch
- Pull requests
- Manual workflow dispatch

**Jobs**:

1. **Check**
   - Runs `cargo check` on all targets
   - Caches dependencies for faster builds

2. **Test Suite**
   - Runs `cargo test --all`
   - Includes unit tests for all pallets
   - Generates and uploads test coverage reports

3. **Clippy**
   - Runs `cargo clippy --all-targets --all-features`
   - Enforces Rust coding standards
   - Fails on warnings

4. **Format**
   - Verifies code formatting with `cargo fmt --all -- --check`
   - Ensures consistent code style

5. **Documentation**
   - Builds Rust documentation
   - Verifies all public items are documented
   - Checks for broken links

### 2. Python CI Workflow

**File**: `.github/workflows/python.yml`

**Triggers**:
- Push to main branch
- Pull requests
- Manual workflow dispatch

**Jobs**:

1. **Lint and Type Check**
   - Runs `black` for code formatting
   - Runs `isort` for import sorting
   - Runs `ruff` for linting
   - Runs `mypy` for type checking
   - Uses UV for dependency management

2. **Test**
   - Sets up Python environment with UV
   - Installs dependencies from `requirements.txt`
   - Runs pytest with coverage reporting
   - Uploads coverage reports to Codecov

### 3. Integration Tests Workflow

**File**: `.github/workflows/integration.yml`

**Triggers**:
- Push to main branch
- Pull requests
- Manual workflow dispatch

**Jobs**:

1. **Integration Tests**
   - Sets up IPFS service container
   - Configures Rust and Python environments
   - Handles submodule checkout
   - Runs combined integration tests
   - Verifies IPFS integration

## Environment Setup

### Required Secrets
- `CODECOV_TOKEN`: Token for coverage report uploads

### Environment Variables
- `IPFS_API_URL`: Default `http://localhost:5001`
- `IPFS_GATEWAY_URL`: Default `http://localhost:8080`

## Local Development

Before pushing changes, run these checks locally:

1. **Rust Checks**
   ```sh
   # Format code
   cargo fmt --all
   # Run clippy
   cargo clippy --all-targets --all-features
   # Run tests
   cargo test --all
   ```

2. **Python Checks**
   ```sh
   # Format code
   black modnet tests
   isort modnet tests
   # Run linters
   ruff check modnet tests
   mypy modnet tests
   # Run tests
   pytest tests/ --cov=modnet
   ```

## Workflow Status Badges

| Workflow | Status |
|----------|--------|
| Rust CI | ![Rust CI](https://github.com/your-org/mod-net/workflows/Rust%20CI/badge.svg) |
| Python CI | ![Python CI](https://github.com/your-org/mod-net/workflows/Python%20CI/badge.svg) |
| Integration Tests | ![Integration Tests](https://github.com/your-org/mod-net/workflows/Integration%20Tests/badge.svg) |

## Adding New Checks

When adding new checks to the CI/CD pipeline:

1. Update the relevant workflow file in `.github/workflows/`
2. Document the changes in this file
3. Update the local development instructions
4. Test the changes in a feature branch
5. Update the project spec if necessary

## Common Issues and Solutions

1. **IPFS Connection Issues**
   - Ensure IPFS daemon is running for local tests
   - Check container logs in CI environment

2. **Python Environment Issues**
   - Use UV for consistent dependency resolution
   - Follow the version constraints in `requirements.in`

3. **Rust Build Cache**
   - Clear local cache if experiencing issues: `cargo clean`
   - CI uses GitHub's cache action for efficiency

## Maintenance

- Review and update dependencies monthly
- Monitor CI/CD performance and optimize as needed
- Update documentation when workflows change
- Archive logs and coverage reports
