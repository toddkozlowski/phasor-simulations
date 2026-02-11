# Phasor Simulations

Python framework for generating publication-quality phasor diagrams. This project creates clean, minimal illustrations of phasor relationships on the complex plane, designed for scientific publications.

## Features

Generate five types of phasor diagrams:

1. **Single phasor** - Basic phasor with rotation indicator
2. **Phasor summation** - Constructive, destructive, and orthogonal interference
3. **Amplitude modulation** - Carrier with sidebands at different time points
4. **Phase modulation** - Carrier with phase variation indicator
5. **Optical cavity fields** - Multiple phasor chains showing interference patterns

All diagrams are exported as both high-resolution PNG (300 DPI) and vector EPS formats.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
python -m uv sync
```

## Usage

Generate all diagrams:

```bash
python -m uv run main.py
```

Or run individual illustration functions from Python:

```python
from main import illustration_1_single_phasor
illustration_1_single_phasor()
```

All generated diagrams are saved to the `images/` directory.

## Customization

The framework is designed for easy modification:

- **Colors and styles**: Edit the color parameters in each function
- **Phasor parameters**: Adjust angles, magnitudes, and positions in the function definitions
- **Circle indicators**: Use `draw_circle()` with custom center and radius
- **Global styles**: Modify `style.json` for consistent styling across all diagrams

## Project Structure

- `main.py` - Main script with all illustration functions
- `style.json` - Style configuration (colors, line widths, fonts)
- `images/` - Output directory for generated diagrams
- `pyproject.toml` - Project dependencies and metadata
