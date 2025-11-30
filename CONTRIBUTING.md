# Contributing to AI Drawing Analyzer

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mohandshamada/AI_Drawings_Reader.git
cd AI_Drawings_Reader
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

For local model support:
```bash
pip install -e ".[local]"
```

### 4. Install Node.js Dependencies (for Toon format)

```bash
pnpm install
# or
npm install
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ai_drawing_analyzer --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

### Code Quality

#### Linting
```bash
# Check code with ruff
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

#### Formatting
```bash
# Check formatting
black --check src/ tests/

# Auto-format
black src/ tests/
```

#### Type Checking
```bash
mypy src/
```

### Running All Checks
```bash
# Run everything at once
ruff check src/ tests/ && \
black --check src/ tests/ && \
mypy src/ && \
pytest --cov=src/ai_drawing_analyzer
```

## Code Style

- **Line Length:** 100 characters
- **Formatting:** Black (automatic)
- **Linting:** Ruff with E, F, I, N, W, UP rules
- **Type Hints:** Encouraged but not required
- **Docstrings:** Required for public APIs (Google style)

### Docstring Example

```python
def process_image(image_path: str, model: str) -> dict:
    """
    Process an image using the specified model.

    Args:
        image_path: Path to the input image file
        model: Model identifier to use for processing

    Returns:
        Dictionary containing processed results with keys:
        - 'text': Extracted text content
        - 'confidence': Confidence score (0-1)

    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If model is not supported

    Example:
        >>> result = process_image('doc.jpg', 'gpt-4o')
        >>> print(result['text'])
    """
    ...
```

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or changes

### 2. Make Your Changes

- Write clear, concise commit messages
- Add tests for new functionality
- Update documentation as needed
- Follow the existing code style

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check code quality
ruff check src/ tests/
black --check src/ tests/
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add support for new model provider

- Add XYZ client implementation
- Add tests for XYZ client
- Update documentation"
```

Commit message format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tooling changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Screenshots (if applicable)
- Test results

## Adding New Features

### Adding a New AI Provider

1. Create a new client in `src/ai_drawing_analyzer/clients/`:

```python
from typing import List, Dict
from .base import BaseClient

class NewProviderClient(BaseClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.newprovider.com"

    async def get_available_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "model-1", "name": "Model 1"},
        ]

    async def analyze_image(self, image_base64: str, prompt: str, model: str) -> str:
        # Implementation
        pass
```

2. Register in `ClientFactory`:

```python
# In factory.py
from .newprovider import NewProviderClient

class ClientFactory:
    PROVIDERS = {
        # ...
        'newprovider': NewProviderClient,
    }

    API_KEY_MAP = {
        # ...
        'newprovider': 'NEWPROVIDER_API_KEY',
    }
```

3. Add tests in `tests/test_clients/test_newprovider.py`

4. Update documentation in README.md

### Adding Tests

Place tests in the appropriate `tests/` subdirectory:

```
tests/
├── test_config.py           # Configuration tests
├── test_pdf_processing.py   # PDF processing tests
└── test_clients/           # Client tests
    ├── test_factory.py
    └── test_newclient.py
```

Use fixtures from `tests/conftest.py`:

```python
def test_analyze_image(sample_image_base64, mock_api_key):
    client = NewClient(mock_api_key)
    result = await client.analyze_image(sample_image_base64, "Test", "model-1")
    assert result is not None
```

## Reporting Issues

### Bug Reports

Include:
- OS and Python version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal code example

### Feature Requests

Include:
- Clear description of the feature
- Use case/motivation
- Proposed implementation (optional)
- Potential alternatives

## Questions?

- Open a GitHub Discussion
- Check existing issues
- Review documentation in `/docs`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
