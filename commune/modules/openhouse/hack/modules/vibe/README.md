# Vibe Module

A module for generating positive vibes through various mediums like quotes, color schemes, and music recommendations.

## Features

- Generate mood-based content for different vibe types
- Get inspirational quotes based on your desired mood
- Get color schemes that match specific vibes
- Get music recommendations that enhance your desired mood

## Vibe Types

The module supports the following vibe types:

- **chill**: Relaxed, laid-back vibes
- **energetic**: High-energy, upbeat vibes
- **creative**: Inspiration for creative thinking
- **focused**: Help with concentration and productivity
- **happy**: Joyful, positive vibes
- **reflective**: Thoughtful, introspective vibes
- **motivated**: Encouragement for achieving goals
- **peaceful**: Calm, serene vibes

## Usage

```python
from commune.modules.openhouse.hack.modules.vibe import Vibe

# Initialize the vibe module
vibe = Vibe()

# Get a random vibe
random_vibe = vibe.get_vibe()
print(random_vibe)

# Get a specific vibe type
chill_vibe = vibe.get_vibe(vibe_type="chill")
print(chill_vibe)

# Get just a quote
quote = vibe.get_quote(vibe_type="motivated")
print(quote)

# Get a color scheme
colors = vibe.get_color_scheme(vibe_type="creative")
print(colors)

# Get a music recommendation
music = vibe.get_music(vibe_type="energetic")
print(music)
```

## Customization

You can customize the module by adding your own quotes, color schemes, and music recommendations in the `data` directory:

- `quotes.json`: Add your own inspirational quotes
- `color_schemes.json`: Define your own color palettes
- `music.json`: Add your own music recommendations

## Example Output

```json
{
  "vibe_type": "chill",
  "quote": "Take it easy, the world will still be here tomorrow.",
  "colors": ["#B3E5FC", "#81D4FA", "#4FC3F7", "#29B6F6", "#03A9F4"],
  "music": {
    "artist": "Tycho",
    "song": "Awake",
    "link": "https://open.spotify.com/track/4jTiyLlOJVJj3mCr7yfPQD"
  },
  "message": "Here's your chill vibe for today!"
}
```