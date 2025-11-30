# Legacy Files

This directory contains legacy/deprecated code that has been superseded by the modular implementation in `src/`.

## Files

### `dwg_analyzer.py`
**Status:** Deprecated
**Replacement:** Use `ai-drawing-analyzer` CLI (installed via `pip install -e .`)

This was the original standalone version that contained all functionality in a single file. It has been replaced by the modular architecture in `src/ai_drawing_analyzer/`.

**Migration:**
```bash
# Old way (deprecated)
python dwg_analyzer.py document.pdf

# New way (recommended)
ai-drawing-analyzer document.pdf
```

The new modular version offers:
- Better code organization and testability
- Type annotations and linting
- Proper package structure
- Async consistency
- Response caching
- Toon format support
- CI/CD integration

This file is kept for reference only and may be removed in future versions.
