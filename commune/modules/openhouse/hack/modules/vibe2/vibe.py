import random
import time
from typing import List, Dict, Any, Optional

class Vibe:
    """
    A class that generates and manages dope vibes.
    """
    
    def __init__(self):
        self.vibes = [
            "chill", "energetic", "dreamy", "nostalgic", "euphoric",
            "groovy", "mellow", "electric", "cosmic", "zen",
            "wavey", "ethereal", "vibrant", "radiant", "mystical"
        ]
        
        self.intensities = ["subtle", "moderate", "strong", "overwhelming", "transcendent"]
        
        self.moods = [
            "relaxed", "inspired", "focused", "blissful", "creative",
            "reflective", "playful", "serene", "passionate", "mindful"
        ]
        
        self.colors = [
            "deep purple", "electric blue", "sunset orange", "neon green", "midnight black",
            "golden yellow", "crimson red", "turquoise", "hot pink", "cosmic indigo"
        ]
    
    def generate_random_vibe(self) -> Dict[str, Any]:
        """
        Generate a random dope vibe with various attributes.
        
        Returns:
            Dict containing vibe attributes
        """
        vibe_type = random.choice(self.vibes)
        intensity = random.choice(self.intensities)
        mood = random.choice(self.moods)
        color = random.choice(self.colors)
        energy_level = random.randint(1, 100)
        
        return {
            "vibe_type": vibe_type,
            "intensity": intensity,
            "mood": mood,
            "color": color,
            "energy_level": energy_level,
            "timestamp": time.time()
        }
    
    def get_vibe_description(self, vibe: Dict[str, Any]) -> str:
        """
        Generate a textual description of a vibe.
        
        Args:
            vibe: A dictionary containing vibe attributes
            
        Returns:
            A string description of the vibe
        """
        return f"A {vibe['intensity']} {vibe['vibe_type']} vibe with {vibe['mood']} energy, radiating {vibe['color']} at level {vibe['energy_level']}"
    
    def match_vibe_to_activity(self, vibe: Dict[str, Any]) -> List[str]:
        """
        Suggest activities that match the given vibe.
        
        Args:
            vibe: A dictionary containing vibe attributes
            
        Returns:
            A list of suggested activities
        """
        activities = []
        
        # Match based on vibe type
        if vibe["vibe_type"] in ["chill", "mellow", "zen"]:
            activities.extend(["meditation", "reading a book", "watching sunset", "gentle yoga"])
        
        elif vibe["vibe_type"] in ["energetic", "electric", "vibrant"]:
            activities.extend(["dancing", "workout", "creative project", "social gathering"])
        
        elif vibe["vibe_type"] in ["dreamy", "nostalgic", "ethereal", "cosmic"]:
            activities.extend(["stargazing", "journaling", "listening to ambient music", "art creation"])
        
        # Match based on mood
        if vibe["mood"] in ["creative", "inspired"]:
            activities.extend(["painting", "writing", "music production", "crafting"])
        
        elif vibe["mood"] in ["relaxed", "serene", "mindful"]:
            activities.extend(["nature walk", "tea ceremony", "stretching", "deep breathing"])
        
        # Return unique activities
        return list(set(activities))[:5]  # Return up to 5 unique activities
    
    def enhance_vibe(self, vibe: Dict[str, Any], factor: float = 1.2) -> Dict[str, Any]:
        """
        Enhance a vibe by increasing its energy level.
        
        Args:
            vibe: A dictionary containing vibe attributes
            factor: Multiplication factor for energy enhancement
            
        Returns:
            Enhanced vibe dictionary
        """
        enhanced_vibe = vibe.copy()
        enhanced_vibe["energy_level"] = min(100, int(enhanced_vibe["energy_level"] * factor))
        
        # Possibly update intensity based on new energy level
        if enhanced_vibe["energy_level"] > 80:
            enhanced_vibe["intensity"] = "transcendent"
        elif enhanced_vibe["energy_level"] > 60:
            enhanced_vibe["intensity"] = "overwhelming"
        elif enhanced_vibe["energy_level"] > 40:
            enhanced_vibe["intensity"] = "strong"
        
        return enhanced_vibe
    
    def mix_vibes(self, vibe1: Dict[str, Any], vibe2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mix two vibes together to create a new hybrid vibe.
        
        Args:
            vibe1: First vibe dictionary
            vibe2: Second vibe dictionary
            
        Returns:
            A new hybrid vibe
        """
        return {
            "vibe_type": f"{vibe1['vibe_type']}-{vibe2['vibe_type']}",
            "intensity": random.choice([vibe1['intensity'], vibe2['intensity']]),
            "mood": f"{vibe1['mood']}-{vibe2['mood']}",
            "color": f"{vibe1['color']} fused with {vibe2['color']}",
            "energy_level": (vibe1["energy_level"] + vibe2["energy_level"]) // 2,
            "timestamp": time.time()
        }


def create_vibe_playlist(vibe_type: str) -> List[str]:
    """
    Generate a playlist of songs that match a specific vibe.
    
    Args:
        vibe_type: The type of vibe to match
        
    Returns:
        A list of song recommendations
    """
    playlists = {
        "chill": [
            "Tycho - Awake",
            "Bonobo - Kerala",
            "Nujabes - Aruarian Dance",
            "Khruangbin - Friday Morning",
            "Four Tet - Lush"
        ],
        "energetic": [
            "Justice - Genesis",
            "The Chemical Brothers - Galvanize",
            "Daft Punk - Around The World",
            "Fatboy Slim - Right Here, Right Now",
            "The Prodigy - Breathe"
        ],
        "dreamy": [
            "Beach House - Space Song",
            "M83 - Midnight City",
            "Tame Impala - Let It Happen",
            "Slowdive - Alison",
            "Cocteau Twins - Cherry-Coloured Funk"
        ],
        "nostalgic": [
            "Mac DeMarco - Chamber of Reflection",
            "Boards of Canada - Roygbiv",
            "Massive Attack - Teardrop",
            "Air - La Femme d'Argent",
            "Portishead - Glory Box"
        ],
        "cosmic": [
            "Flying Lotus - Zodiac Shit",
            "Floating Points - Silhouettes",
            "Jon Hopkins - Emerald Rush",
            "Aphex Twin - Xtal",
            "Solar Fields - Discovering"
        ]
    }
    
    # Return default playlist if vibe type not found
    return playlists.get(vibe_type.lower(), playlists["chill"])


def get_vibe_quote() -> str:
    """
    Return a random quote about vibes and energy.
    
    Returns:
        A string containing an inspirational quote
    """
    quotes = [
        "Vibrate on your own frequency.",
        "Your vibe attracts your tribe.",
        "Good vibes only.",
        "Raise your vibration to change your life.",
        "Energy flows where attention goes.",
        "The universe responds to your frequency.",
        "Be the energy you want to attract.",
        "High vibes heal lives.",
        "Tune into the frequency of your dreams.",
        "Protect your peace, elevate your vibe."
    ]
    
    return random.choice(quotes)
