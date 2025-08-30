import random
import json
import os
from typing import List, Dict, Any, Optional

class Vibe:
    """
    A module for generating positive vibes through various mediums.
    
    This class provides functionality to create uplifting experiences through
    music recommendations, quotes, color schemes, and more.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Vibe module with optional configuration."""
        self.config = config or {}
        self.vibe_types = [
            "chill", "energetic", "creative", "focused", 
            "happy", "reflective", "motivated", "peaceful"
        ]
        
        # Load resources
        self.quotes = self._load_quotes()
        self.color_schemes = self._load_color_schemes()
        self.music_recommendations = self._load_music_recommendations()
        
    def _load_quotes(self) -> Dict[str, List[str]]:
        """Load positive quotes for different vibes."""
        # Default quotes in case file doesn't exist
        default_quotes = {
            "chill": [
                "Take it easy, the world will still be here tomorrow.",
                "Breathe in peace, breathe out stress.",
                "The calm mind is the ultimate weapon against your challenges."
            ],
            "energetic": [
                "Your energy introduces you before you even speak.",
                "Energy and persistence conquer all things.",
                "Good energy is contagious."
            ],
            "creative": [
                "Creativity is intelligence having fun.",
                "Create the things you wish existed.",
                "The creative adult is the child who survived."
            ],
            "focused": [
                "Where focus goes, energy flows.",
                "Stay focused, go after your dreams, and keep moving toward your goals.",
                "Focus on the journey, not the destination."
            ],
            "happy": [
                "Happiness is not by chance, but by choice.",
                "The most wasted of days is one without laughter.",
                "Do more of what makes you happy."
            ],
            "reflective": [
                "Life can only be understood backwards, but it must be lived forwards.",
                "Reflect on your present blessings, not on your past misfortunes.",
                "Sometimes you need to step outside, clear your head and remind yourself of who you are."
            ],
            "motivated": [
                "Your only limit is you.",
                "Don't stop when you're tired. Stop when you're done.",
                "The harder you work for something, the greater you'll feel when you achieve it."
            ],
            "peaceful": [
                "Peace comes from within. Do not seek it without.",
                "Peace is the result of retraining your mind to process life as it is.",
                "When you do what you fear most, then you can do anything."
            ]
        }
        
        quotes_path = os.path.join(os.path.dirname(__file__), 'data', 'quotes.json')
        try:
            if os.path.exists(quotes_path):
                with open(quotes_path, 'r') as f:
                    return json.load(f)
            return default_quotes
        except Exception:
            return default_quotes
    
    def _load_color_schemes(self) -> Dict[str, List[str]]:
        """Load color schemes for different vibes."""
        # Default color schemes (hex codes)
        default_schemes = {
            "chill": ["#B3E5FC", "#81D4FA", "#4FC3F7", "#29B6F6", "#03A9F4"],
            "energetic": ["#FF5252", "#FF8A80", "#FF1744", "#D50000", "#FF4081"],
            "creative": ["#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC", "#9C27B0"],
            "focused": ["#C5CAE9", "#9FA8DA", "#7986CB", "#5C6BC0", "#3F51B5"],
            "happy": ["#FFF59D", "#FFF176", "#FFEE58", "#FFEB3B", "#FDD835"],
            "reflective": ["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3"],
            "motivated": ["#FFCCBC", "#FFAB91", "#FF8A65", "#FF7043", "#FF5722"],
            "peaceful": ["#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#4CAF50"]
        }
        
        schemes_path = os.path.join(os.path.dirname(__file__), 'data', 'color_schemes.json')
        try:
            if os.path.exists(schemes_path):
                with open(schemes_path, 'r') as f:
                    return json.load(f)
            return default_schemes
        except Exception:
            return default_schemes
    
    def _load_music_recommendations(self) -> Dict[str, List[Dict[str, str]]]:
        """Load music recommendations for different vibes."""
        # Default music recommendations
        default_recommendations = {
            "chill": [
                {"artist": "Tycho", "song": "Awake", "link": "https://open.spotify.com/track/4jTiyLlOJVJj3mCr7yfPQD"},
                {"artist": "Bonobo", "song": "Kerala", "link": "https://open.spotify.com/track/1hdxVWPEmDrBguXpwURm85"},
                {"artist": "Nujabes", "song": "Feather", "link": "https://open.spotify.com/track/2ej1A2Ze6P2EOW7KfIosZR"}
            ],
            "energetic": [
                {"artist": "The Prodigy", "song": "Breathe", "link": "https://open.spotify.com/track/5zVOkZP92qfw0WnXXKHwDl"},
                {"artist": "Chemical Brothers", "song": "Galvanize", "link": "https://open.spotify.com/track/0Qw8jJnYm6tIX7Yl5pEbUz"},
                {"artist": "Daft Punk", "song": "Harder, Better, Faster, Stronger", "link": "https://open.spotify.com/track/5W3cjX2J3tjhG8zb6u0qHn"}
            ],
            "creative": [
                {"artist": "Flying Lotus", "song": "Never Catch Me", "link": "https://open.spotify.com/track/2Nt4Uw91pQLXSJ28SttDdF"},
                {"artist": "Radiohead", "song": "Everything In Its Right Place", "link": "https://open.spotify.com/track/2yPr9G3glBCLzFqaVJeRlR"},
                {"artist": "Björk", "song": "Hyperballad", "link": "https://open.spotify.com/track/2KgW0pUQEJfK4yyKWdNCXX"}
            ],
            "focused": [
                {"artist": "Brian Eno", "song": "1/1", "link": "https://open.spotify.com/track/7M2tXmeS15NAP8E8hJ97wd"},
                {"artist": "Max Richter", "song": "On The Nature Of Daylight", "link": "https://open.spotify.com/track/1BSMpVGWs3v5BZKnAQziAc"},
                {"artist": "Nils Frahm", "song": "Says", "link": "https://open.spotify.com/track/1wHZx0LgzFHyeIZkUuRzk9"}
            ],
            "happy": [
                {"artist": "Pharrell Williams", "song": "Happy", "link": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCO"},
                {"artist": "Earth, Wind & Fire", "song": "September", "link": "https://open.spotify.com/track/2grjqo0Frpf2okIBiifQKs"},
                {"artist": "Daft Punk", "song": "Get Lucky", "link": "https://open.spotify.com/track/2Foc5Q5nqNiosCNqttzHof"}
            ],
            "reflective": [
                {"artist": "Bon Iver", "song": "Holocene", "link": "https://open.spotify.com/track/4fbvXwMi4fL4EgSvLf1HJr"},
                {"artist": "Sigur Rós", "song": "Hoppípolla", "link": "https://open.spotify.com/track/6eTGxxQxiTFE6LfZHC33Wm"},
                {"artist": "Olafur Arnalds", "song": "Near Light", "link": "https://open.spotify.com/track/0Qg5iMtYtvayMJQWUkGsCu"}
            ],
            "motivated": [
                {"artist": "Kendrick Lamar", "song": "DNA.", "link": "https://open.spotify.com/track/6HZILIRieu8S0iqY8kIKhj"},
                {"artist": "Kanye West", "song": "Power", "link": "https://open.spotify.com/track/2gZUPNdnz5Y45eiGxpHGSc"},
                {"artist": "Run The Jewels", "song": "Legend Has It", "link": "https://open.spotify.com/track/6jHG1YQkqgojdEzm8XgCnx"}
            ],
            "peaceful": [
                {"artist": "Hammock", "song": "Breathturn", "link": "https://open.spotify.com/track/7xtmJBut9ziVHW00enQxSD"},
                {"artist": "Stars of the Lid", "song": "Requiem for Dying Mothers", "link": "https://open.spotify.com/track/1mhJaYAFgAp7GEQh8RM4JO"},
                {"artist": "Eluvium", "song": "Radio Ballet", "link": "https://open.spotify.com/track/65NwOZqoXny4JxqAPlfxRD"}
            ]
        }
        
        music_path = os.path.join(os.path.dirname(__file__), 'data', 'music.json')
        try:
            if os.path.exists(music_path):
                with open(music_path, 'r') as f:
                    return json.load(f)
            return default_recommendations
        except Exception:
            return default_recommendations
    
    def get_vibe(self, vibe_type: Optional[str] = None) -> Dict[str, Any]:
        """Get a complete vibe package with quotes, colors, and music.
        
        Args:
            vibe_type: The type of vibe to generate. If None, a random vibe is selected.
            
        Returns:
            A dictionary containing the vibe package with quotes, colors, and music.
        """
        if vibe_type is None:
            vibe_type = random.choice(self.vibe_types)
        elif vibe_type not in self.vibe_types:
            raise ValueError(f"Vibe type must be one of {self.vibe_types}")
        
        quote = random.choice(self.quotes.get(vibe_type, ["Good vibes only"]))
        colors = self.color_schemes.get(vibe_type, ["#000000"])
        music = random.choice(self.music_recommendations.get(vibe_type, [{"artist": "Unknown", "song": "Unknown"}]))
        
        return {
            "vibe_type": vibe_type,
            "quote": quote,
            "colors": colors,
            "music": music,
            "message": f"Here's your {vibe_type} vibe for today!"
        }
    
    def get_quote(self, vibe_type: Optional[str] = None) -> str:
        """Get a random positive quote for a specific vibe type.
        
        Args:
            vibe_type: The type of vibe to get a quote for. If None, a random vibe is selected.
            
        Returns:
            A string containing a positive quote.
        """
        if vibe_type is None:
            vibe_type = random.choice(list(self.quotes.keys()))
        elif vibe_type not in self.quotes:
            return "Good vibes only!"
        
        return random.choice(self.quotes[vibe_type])
    
    def get_color_scheme(self, vibe_type: Optional[str] = None) -> List[str]:
        """Get a color scheme for a specific vibe type.
        
        Args:
            vibe_type: The type of vibe to get colors for. If None, a random vibe is selected.
            
        Returns:
            A list of color hex codes.
        """
        if vibe_type is None:
            vibe_type = random.choice(list(self.color_schemes.keys()))
        elif vibe_type not in self.color_schemes:
            return ["#000000"]
        
        return self.color_schemes[vibe_type]
    
    def get_music(self, vibe_type: Optional[str] = None) -> Dict[str, str]:
        """Get a music recommendation for a specific vibe type.
        
        Args:
            vibe_type: The type of vibe to get music for. If None, a random vibe is selected.
            
        Returns:
            A dictionary containing artist, song, and link information.
        """
        if vibe_type is None:
            vibe_type = random.choice(list(self.music_recommendations.keys()))
        elif vibe_type not in self.music_recommendations:
            return {"artist": "Unknown", "song": "Unknown", "link": ""}
        
        return random.choice(self.music_recommendations[vibe_type])
    
    def forward(self, vibe_type: Optional[str] = None) -> Dict[str, Any]:
        """Default forward method that returns a complete vibe package.
        
        Args:
            vibe_type: The type of vibe to generate. If None, a random vibe is selected.
            
        Returns:
            A dictionary containing the vibe package with quotes, colors, and music.
        """
        return self.get_vibe(vibe_type)
