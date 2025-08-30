import random
import time
from typing import List, Dict, Any, Optional
from .vibe import Vibe, create_vibe_playlist, get_vibe_quote

class VibeGenerator:
    """
    A class that generates and manages dope vibes with additional features.
    """
    
    def __init__(self):
        self.vibe_engine = Vibe()
        self.session_vibes = []
        self.favorite_vibes = []
    
    def generate_vibe(self) -> Dict[str, Any]:
        """
        Generate a new vibe and add it to the session history.
        
        Returns:
            A dictionary containing vibe attributes
        """
        new_vibe = self.vibe_engine.generate_random_vibe()
        self.session_vibes.append(new_vibe)
        return new_vibe
    
    def get_current_vibe(self) -> Dict[str, Any]:
        """
        Get the most recently generated vibe.
        
        Returns:
            The most recent vibe or a new one if none exists
        """
        if not self.session_vibes:
            return self.generate_vibe()
        return self.session_vibes[-1]
    
    def save_favorite(self, vibe: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a vibe to favorites.
        
        Args:
            vibe: The vibe to save (defaults to current vibe)
        """
        if vibe is None:
            vibe = self.get_current_vibe()
        self.favorite_vibes.append(vibe)
    
    def get_vibe_report(self, vibe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive report about a vibe.
        
        Args:
            vibe: The vibe to report on (defaults to current vibe)
            
        Returns:
            A dictionary with detailed vibe information
        """
        if vibe is None:
            vibe = self.get_current_vibe()
            
        description = self.vibe_engine.get_vibe_description(vibe)
        activities = self.vibe_engine.match_vibe_to_activity(vibe)
        playlist = create_vibe_playlist(vibe["vibe_type"])
        quote = get_vibe_quote()
        
        return {
            "vibe": vibe,
            "description": description,
            "recommended_activities": activities,
            "playlist": playlist,
            "inspirational_quote": quote,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    
    def create_custom_vibe(self, vibe_type: str, mood: str, color: str, 
                         energy_level: int) -> Dict[str, Any]:
        """
        Create a custom vibe with specified attributes.
        
        Args:
            vibe_type: Type of vibe
            mood: Mood associated with the vibe
            color: Color associated with the vibe
            energy_level: Energy level (1-100)
            
        Returns:
            A custom vibe dictionary
        """
        intensity = "subtle"
        if energy_level > 80:
            intensity = "transcendent"
        elif energy_level > 60:
            intensity = "overwhelming"
        elif energy_level > 40:
            intensity = "strong"
        elif energy_level > 20:
            intensity = "moderate"
            
        custom_vibe = {
            "vibe_type": vibe_type,
            "intensity": intensity,
            "mood": mood,
            "color": color,
            "energy_level": max(1, min(100, energy_level)),
            "timestamp": time.time()
        }
        
        self.session_vibes.append(custom_vibe)
        return custom_vibe
    
    def get_vibe_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of vibes generated in this session.
        
        Returns:
            List of vibe dictionaries
        """
        return self.session_vibes
    
    def get_favorite_vibes(self) -> List[Dict[str, Any]]:
        """
        Get the list of favorite vibes.
        
        Returns:
            List of favorite vibe dictionaries
        """
        return self.favorite_vibes
    
    def clear_session(self) -> None:
        """
        Clear the current session history.
        """
        self.session_vibes = []
    
    def vibe_evolution(self, steps: int = 5) -> List[Dict[str, Any]]:
        """
        Generate an evolution of vibes, each building on the previous.
        
        Args:
            steps: Number of evolution steps
            
        Returns:
            List of evolving vibes
        """
        evolution = []
        current_vibe = self.vibe_engine.generate_random_vibe()
        evolution.append(current_vibe)
        
        for _ in range(steps - 1):
            factor = 1.0 + (random.random() * 0.4 - 0.2)  # Random factor between 0.8 and 1.2
            current_vibe = self.vibe_engine.enhance_vibe(current_vibe, factor)
            evolution.append(current_vibe)
            
        return evolution