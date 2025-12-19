# CI/CD Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipelines configured for the Tensorax project.

## Overview

Tensorax uses GitHub Actions for automated testing, code quality checks, and deployment. The CI/CD pipeline ensures code quality, test coverage, and compatibility across different platforms and Python versions.

## Workflows

### 1. Tests Workflow (`tests.yml`)

**Triggers:**

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

**Jobs:**

#### Test Job

- **Matrix Strategy:** Tests across multiple OS and Python versions
  - Operating Systems: Ubuntu, macOS, Windows
  - Python Versions: 3.8, 3.9, 3.10, 3.11, 3.12
- **Steps:**
  1. Checkout code
  2. Set up Python environment with caching
  3. Install system dependencies (OS-specific)
  4. Install Python dependencies
  5. Build C++ extension
  6. Run tests with coverage
  7. Upload coverage to Codecov (Ubuntu + Python 3.12 only)

#### CUDA Test Job

- **Environment:** NVIDIA CUDA 11.8 Docker container
- **Purpose:** Verify CUDA build compatibility
- **Steps:**
  1. Checkout code
  2. Install system dependencies
  3. Build C++ extension with CUDA support
  4. Run tests (CPU mode in CI environment)

#### Lint Job

- **Code Quality Checks:**
  - Black: Code formatting
  - isort: Import sorting
  - flake8: Linting and style guide enforcement
- **Purpose:** Ensure consistent code style

#### Docs Job

- **Purpose:** Verify documentation files exist and are valid
- **Checks:**
  - All documentation files present
  - Links in README are valid

#### Build Job

- **Purpose:** Build distribution packages
- **Artifacts:** Python wheel and source distribution
- **Steps:**
  1. Build sdist and wheel
  2. Check distribution with twine
  3. Upload artifacts for 7 days

#### Status Job

- **Purpose:** Summarize all job results
- **Dependencies:** All other jobs
- **Outcome:** Overall CI status

**Status Badge:**

```markdown
[![CI](https://github.com/NotShrirang/tensorax/workflows/Tests/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/tests.yml)
```

---

### 2. Code Coverage Workflow (`coverage.yml`)

**Triggers:**

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

**Purpose:** Generate and track code coverage metrics

**Steps:**

1. Run full test suite with coverage
2. Generate HTML, XML, and terminal coverage reports
3. Upload coverage to Codecov
4. Upload HTML report as artifact
5. Display coverage summary in GitHub Actions summary

**Artifacts:**

- HTML coverage report (30 days retention)
- XML coverage for Codecov integration

**Status Badge:**

```markdown
[![Code Coverage](https://github.com/NotShrirang/tensorax/workflows/Code%20Coverage/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/coverage.yml)
```

---

### 3. Performance Benchmarks Workflow (`benchmark.yml`)

**Triggers:**

- Push to `main` branch
- Pull requests to `main` branch
- Weekly schedule (Monday at 00:00 UTC)
- Manual workflow dispatch

**Purpose:** Track performance metrics over time

**Benchmarks:**

- Matrix multiplication (various sizes: 64×64 to 1024×1024)
- Element-wise operations (1024×1024)
- Reduction operations (1000×1000)

**Artifacts:**

- Benchmark results (30 days retention)

---

### 4. Publish to PyPI Workflow (`publish.yml`)

**Triggers:**

- GitHub release published
- Manual workflow dispatch (for Test PyPI)

**Purpose:** Automated package publishing

**Steps:**

1. Build distribution packages
2. Validate with twine
3. Publish to PyPI (on release) or Test PyPI (manual)

**Required Secrets:**

- `PYPI_API_TOKEN`: PyPI API token for production releases
- `TEST_PYPI_API_TOKEN`: Test PyPI token for testing

---

## Setup Instructions

### 1. Enable GitHub Actions

GitHub Actions are automatically enabled for repositories. Workflows will run on push/PR events.

### 2. Branch Protection Rules

**Recommended settings for `main` branch:**

1. Go to **Settings → Branches → Add rule**
2. Branch name pattern: `main`
3. Enable:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date
   - ✅ Status checks: `Test`, `Lint`, `Docs`, `Build`
   - ✅ Require pull request reviews (1 reviewer)
   - ✅ Dismiss stale reviews on new commits
   - ✅ Require signed commits (optional)

---

## Monitoring and Maintenance

### Viewing Workflow Results

1. Navigate to **Actions** tab in GitHub repository
2. Click on workflow name to see runs
3. Click on specific run to view details
4. Download artifacts from successful runs

### Troubleshooting Failed Workflows

#### Test Failures

1. Click on failed job
2. Expand failed step
3. Review error messages and logs
4. Fix issues locally and push again

#### Build Failures

- Check compiler errors in build step
- Verify system dependencies are installed
- Ensure Python version compatibility

#### Coverage Drops

- Review coverage report artifact
- Identify untested code paths
- Add tests for uncovered lines

### Performance Tracking

View benchmark results:

1. Go to **Actions** → **Performance Benchmarks**
2. Download benchmark results artifact
3. Compare across runs to track performance trends

---

## Local Testing

Before pushing, test locally to catch issues early:

```bash
# Run full test suite
pytest tests/ -v

# Check code formatting
black --check tensorax/ tests/

# Check import sorting
isort --check-only tensorax/ tests/

# Lint code
flake8 tensorax/ tests/

# Generate coverage report
pytest tests/ --cov=tensorax --cov-report=html

# Build distribution
python -m build

# Check distribution
twine check dist/*
```

---

## Workflow Status

| Workflow | Status                                                                                                                                                             | Purpose                       |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| Tests    | [![Tests](https://github.com/NotShrirang/tensorax/workflows/Tests/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/tests.yml)                 | Core testing across platforms |
| Coverage | [![Coverage](https://github.com/NotShrirang/tensorax/workflows/Code%20Coverage/badge.svg)](https://github.com/NotShrirang/tensorax/actions/workflows/coverage.yml) | Code coverage tracking        |
| Publish  | [![Publish](https://img.shields.io/pypi/v/tensorax.svg)](https://pypi.org/project/tensorax/)                                                                       | Package publishing            |

---

## Best Practices

### For Contributors

1. **Always run tests locally** before pushing
2. **Ensure code is formatted** with Black
3. **Add tests** for new features
4. **Update documentation** as needed
5. **Keep PRs focused** on single features/fixes
6. **Wait for CI** to pass before requesting review

### For Maintainers

1. **Review CI results** before merging PRs
2. **Monitor coverage trends** over time
3. **Update dependencies** regularly
4. **Review benchmark results** for performance regressions
5. **Use semantic versioning** for releases
6. **Test releases** on Test PyPI before production

---

## Future Enhancements

Planned CI/CD improvements:

- [ ] Automated security scanning (CodeQL, Dependabot)
- [ ] Automated changelog generation
- [ ] Performance regression detection with alerts
- [ ] Docker image building and publishing
- [ ] Integration with cloud providers for GPU testing
- [ ] Automated documentation deployment
- [ ] Nightly builds with latest dependencies
- [ ] Pre-commit hooks automation

---

## Support

For CI/CD issues:

- Check workflow logs in GitHub Actions
- Review this documentation
- Open an issue with `ci/cd` label
- Contact maintainers

**Last Updated:** December 9, 2025
