# Dope Vibe Generator

A Python module for generating, managing, and visualizing dope vibes. This module provides tools to create random vibes, customize them, generate playlists, and get activity recommendations based on the current vibe.

## Features

- Generate random vibes with various attributes (type, intensity, mood, color, energy level)
- Create custom vibes with specific characteristics
- Visualize vibes with ASCII art and color
- Get activity recommendations based on the current vibe
- Generate curated playlists that match specific vibes
- Create vibe evolution sequences
- Get inspirational quotes related to vibes and energy

## Installation

Clone this repository and install the package:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

The module comes with a command-line interface for easy interaction:

```bash
# Generate a random vibe
python -m vibe2.cli generate

# Generate a comprehensive vibe report
python -m vibe2.cli report

# Show an evolution of vibes
python -m vibe2.cli evolve --steps 7

# Create a custom vibe
python -m vibe2.cli custom --type cosmic --mood creative --color "electric blue" --energy 85

# Show a random vibe quote
python -m vibe2.cli quote

# Show a playlist for a specific vibe
python -m vibe2.cli playlist --type chill
```

### Python API

You can also use the module programmatically in your Python code:

```python
from vibe2 import Vibe, VibeGenerator, VibeVisualizer

# Create a vibe generator
generator = VibeGenerator()

# Generate a random vibe
vibe = generator.generate_vibe()
print(vibe)

# Get a comprehensive vibe report
report = generator.get_vibe_report(vibe)

# Visualize the vibe
visualizer = VibeVisualizer()
print(visualizer.visualize_vibe(vibe))

# Create a custom vibe
custom_vibe = generator.create_custom_vibe(
    vibe_type="cosmic",
    mood="creative",
    color="electric blue",
    energy_level=85
)

# Generate a vibe evolution
evolution = generator.vibe_evolution(steps=5)
print(visualizer.visualize_vibe_evolution(evolution))
```

## Module Structure

- `vibe.py`: Core vibe functionality and data
- `vibe_generator.py`: Advanced vibe generation and management
- `visualizer.py`: Text-based visualization for vibes
- `cli.py`: Command-line interface

## Example Output

```
########################################
# VIBE: COSMIC                        #
# ✧･ﾟ: *✧･ﾟ✧･ﾟ: *✧･ﾟ✧･ﾟ: *✧･ﾟ         #
# MOOD: CREATIVE | INTENSITY: STRONG  #
# ENERGY: 75/100                      #
########################################

Description: A strong cosmic vibe with creative energy, radiating electric blue at level 75

RECOMMENDED ACTIVITIES:
  1. stargazing
  2. painting
  3. music production
  4. art creation
  5. journaling

PLAYLIST:
  1. Flying Lotus - Zodiac Shit
  2. Floating Points - Silhouettes
  3. Jon Hopkins - Emerald Rush
  4. Aphex Twin - Xtal
  5. Solar Fields - Discovering

QUOTE: "Tune into the frequency of your dreams."
```

## License

MIT License