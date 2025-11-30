# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - claude-Manjaro Branch

### Added
- **Testing Infrastructure**
  - Complete pytest test suite with fixtures
  - Unit tests for PDF processing, configuration, and client factory
  - Code coverage reporting
  - Test configuration in pyproject.toml

- **Code Quality Tools**
  - Ruff linter configuration
  - Black code formatter configuration
  - MyPy type checking setup
  - Pre-configured dev dependencies

- **Toon Format Support**
  - Node.js bridge for @toon-format/toon package
  - `--to-toon` CLI flag for automatic conversion
  - ToonConverter class for Python-to-Node.js integration
  - Conversion script in `scripts/convert_to_toon.mjs`

- **Response Caching**
  - `ResponseCache` utility class for caching API responses
  - SHA-256 based cache keys
  - Optional caching to reduce redundant API calls
  - Cache management (clear, size)

- **CI/CD Pipeline**
  - GitHub Actions workflow for automated testing
  - Multi-version Python testing (3.9-3.13)
  - Codecov integration for coverage reporting
  - Automated linting and formatting checks

- **Documentation**
  - Comprehensive docstrings for BaseClient, ClientFactory
  - Legacy code documentation in `legacy/README.md`
  - Enhanced inline documentation
  - Usage examples in docstrings

### Changed
- **Async Improvements**
  - Fixed async consistency in HuggingFaceLocalClient
  - Local inference now runs in ThreadPoolExecutor for non-blocking operation
  - Proper async/await throughout codebase

- **Logging Improvements**
  - Replaced all `print()` statements with logger calls
  - Consistent logging in files.py and huggingface.py
  - Better log messages for downloads and model loading

- **Error Handling**
  - Specific exception handlers for network, timeout, and file errors
  - Better error messages with context
  - Debug mode for stack traces (enable with DEBUG env var)

- **Code Organization**
  - Moved `dwg_analyzer.py` to `legacy/` directory
  - Added proper package structure documentation
  - Improved imports and module organization

### Fixed
- Input parameter typo in HuggingFace local client (input_id → input_ids)
- Missing httpx import in cli.py for exception handling

### Infrastructure
- Added `.github/workflows/test.yml` for CI/CD
- Updated `.gitignore` for Node.js, coverage, and cache files
- Added `package.json` for Node.js dependencies
- Enhanced `.gitignore` patterns

## [2.2.0] - 2025-11-28

### Added
- Resume processing capability
- Configuration management with environment variables
- Enhanced error handling and validation
- Output format options

### Changed
- Complete README rewrite
- Improved CLI interface
- Better documentation

## [2.0.0] - Initial Modular Release

### Added
- Modular architecture with separate clients
- Support for 6 AI providers
- PDF to image processing
- JSONL output format
- Text converter for LLM-ready output
