import random
import time
from typing import List, Dict, Any, Optional

class Dopeness:
    """A class that brings dope vibes to your project."""
    
    def __init__(self, vibe_level: int = 10, chill_factor: float = 0.8):
        """Initialize the Dopeness with customizable vibe levels.
        
        Args:
            vibe_level: How dope the vibes should be (1-10)
            chill_factor: How chill the vibes are (0.0-1.0)
        """
        self.vibe_level = min(max(1, vibe_level), 10)  # Keep it between 1-10
        self.chill_factor = min(max(0.0, chill_factor), 1.0)  # Keep it between 0.0-1.0
        self.vibe_phrases = [
            "That's fire", "Straight vibin'", "Too cool for school",
            "Absolute mood", "Chill AF", "Big energy", "No cap",
            "It's lit", "Vibe check passed", "Dope as heck",
            "Straight up good times", "Legendary status"
        ]
        
    def generate_vibe(self) -> str:
        """Generate a random vibe phrase based on the current settings."""
        selected_phrases = random.sample(
            self.vibe_phrases, 
            k=min(self.vibe_level, len(self.vibe_phrases))
        )
        return random.choice(selected_phrases)
    
    def vibe_check(self, input_vibe: str) -> Dict[str, Any]:
        """Check if a given input passes the vibe check.
        
        Args:
            input_vibe: The vibe to check
            
        Returns:
            Dict with vibe check results
        """
        # The longer the input, the more likely it is to pass the vibe check
        length_factor = min(len(input_vibe) / 50, 1.0)
        
        # Random factor influenced by chill_factor
        random_factor = random.random() * self.chill_factor
        
        # Calculate vibe score
        vibe_score = (length_factor + random_factor) * 10
        
        return {
            "input": input_vibe,
            "vibe_score": round(vibe_score, 2),
            "passed": vibe_score >= 5.0,
            "message": self.generate_vibe() if vibe_score >= 5.0 else "Not vibin' with this"
        }
    
    def chill_out(self, duration: float = 1.0) -> str:
        """Take a chill moment.
        
        Args:
            duration: How long to chill for (in seconds)
            
        Returns:
            A chill message
        """
        actual_duration = duration * self.chill_factor
        time.sleep(actual_duration)
        return f"Chilled for {actual_duration:.2f} seconds. Feeling {self.generate_vibe().lower()}."
    
    def enhance_vibes(self, text: str) -> str:
        """Enhance any text with dope vibes.
        
        Args:
            text: The text to enhance
            
        Returns:
            Enhanced text with dope vibes
        """
        enhancements = [
            "✨", "🔥", "💯", "😎", "🙌", "⚡", "🌊", "🤙", "🎵", "🎧"
        ]
        
        # Add random enhancements based on vibe level
        enhanced_text = text
        for _ in range(self.vibe_level):
            position = random.randint(0, len(enhanced_text))
            enhancement = random.choice(enhancements)
            enhanced_text = enhanced_text[:position] + " " + enhancement + " " + enhanced_text[position:]
            
        return enhanced_text
    
    def vibe_playlist(self, mood: Optional[str] = None) -> List[str]:
        """Generate a playlist that matches the vibe.
        
        Args:
            mood: Optional specific mood to match
            
        Returns:
            List of song titles that match the vibe
        """
        chill_tracks = [
            "Lofi Study Mix",
            "Chill Hop Essentials",
            "Bedroom Pop Vibes",
            "Mellow Acoustics",
            "Rainy Day Jazz"
        ]
        
        hype_tracks = [
            "Ultimate Workout Mix",
            "Party Anthems 2023",
            "Top 40 Remix",
            "EDM Festival Hits",
            "Rap Bangers"
        ]
        
        if mood == "chill" or (not mood and self.chill_factor > 0.6):
            return random.sample(chill_tracks, k=min(self.vibe_level, len(chill_tracks)))
        elif mood == "hype" or (not mood and self.chill_factor <= 0.6):
            return random.sample(hype_tracks, k=min(self.vibe_level, len(hype_tracks)))
        else:
            # Mix of both
            all_tracks = chill_tracks + hype_tracks
            return random.sample(all_tracks, k=min(self.vibe_level, len(all_tracks)))
