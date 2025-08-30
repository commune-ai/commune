import random
import time
from typing import Dict, Any, List, Optional

class VibeVisualizer:
    """
    A class that provides text-based visualizations for vibes.
    """
    
    def __init__(self):
        self.ascii_patterns = {
            "chill": [
                "~~~~~",
                "≈≈≈≈≈",
                "∽∽∽∽∽",
                "≋≋≋≋≋",
                "≈~≈~≈"
            ],
            "energetic": [
                "*****",
                "★★★★★",
                "⚡⚡⚡⚡⚡",
                "✧✧✧✧✧",
                "✦✦✦✦✦"
            ],
            "dreamy": [
                "○○○○○",
                "◌◌◌◌◌",
                "◦◦◦◦◦",
                "◯◯◯◯◯",
                "⊙⊙⊙⊙⊙"
            ],
            "cosmic": [
                "✧･ﾟ: *✧･ﾟ",
                "⋆｡°✩ ⋆｡°✩",
                "✦✧✦✧✦",
                "⊹⊹⊹⊹⊹",
                "⋆⋆⋆⋆⋆"
            ],
            "zen": [
                "┌─┐  ┌─┐",
                "└┬┘  └┬┘",
                " │    │ ",
                "┌┴─  ─┴┐",
                "└───────┘"
            ]
        }
        
        self.color_codes = {
            "deep purple": "\033[95m",
            "electric blue": "\033[94m",
            "sunset orange": "\033[91m",
            "neon green": "\033[92m",
            "midnight black": "\033[90m",
            "golden yellow": "\033[93m",
            "crimson red": "\033[31m",
            "turquoise": "\033[96m",
            "hot pink": "\033[95m",
            "cosmic indigo": "\033[35m"
        }
        
        self.reset_color = "\033[0m"
    
    def get_pattern_for_vibe(self, vibe: Dict[str, Any]) -> str:
        """
        Get an ASCII pattern representing the vibe.
        
        Args:
            vibe: A dictionary containing vibe attributes
            
        Returns:
            ASCII art pattern representing the vibe
        """
        vibe_type = vibe["vibe_type"]
        
        # Handle compound vibe types by splitting
        if "-" in vibe_type:
            parts = vibe_type.split("-")
            if parts[0] in self.ascii_patterns:
                vibe_type = parts[0]
            elif parts[1] in self.ascii_patterns:
                vibe_type = parts[1]
        
        # Default to chill if not found
        patterns = self.ascii_patterns.get(vibe_type, self.ascii_patterns["chill"])
        return random.choice(patterns)
    
    def get_color_code(self, color: str) -> str:
        """
        Get ANSI color code for a given color name.
        
        Args:
            color: Color name
            
        Returns:
            ANSI color code
        """
        # Handle compound colors by taking the first one
        if " fused with " in color:
            color = color.split(" fused with ")[0]
            
        return self.color_codes.get(color, self.reset_color)
    
    def visualize_vibe(self, vibe: Dict[str, Any]) -> str:
        """
        Create a text-based visualization of a vibe.
        
        Args:
            vibe: A dictionary containing vibe attributes
            
        Returns:
            String containing a text visualization
        """
        pattern = self.get_pattern_for_vibe(vibe)
        color_code = self.get_color_code(vibe["color"])
        reset = self.reset_color
        
        energy = vibe["energy_level"]
        intensity = vibe["intensity"]
        mood = vibe["mood"]
        
        # Create a frame based on energy level
        frame_char = "·" if energy < 30 else "-" if energy < 60 else "=" if energy < 90 else "#"
        frame_width = 40
        
        # Build the visualization
        lines = []
        lines.append(f"{frame_char * frame_width}")
        lines.append(f"{frame_char} {color_code}VIBE: {vibe['vibe_type'].upper()}{reset} {frame_char}")
        lines.append(f"{frame_char} {color_code}{pattern * 3}{reset} {frame_char}")
        lines.append(f"{frame_char} MOOD: {mood.upper()} | INTENSITY: {intensity.upper()} {frame_char}")
        lines.append(f"{frame_char} ENERGY: {energy}/100 {frame_char}")
        lines.append(f"{frame_char * frame_width}")
        
        return "\n".join(lines)
    
    def visualize_vibe_evolution(self, vibes: List[Dict[str, Any]]) -> str:
        """
        Visualize the evolution of vibes over time.
        
        Args:
            vibes: List of vibe dictionaries in chronological order
            
        Returns:
            String visualization of vibe evolution
        """
        if not vibes:
            return "No vibes to visualize"
            
        lines = ["VIBE EVOLUTION:"]
        lines.append("=" * 40)
        
        for i, vibe in enumerate(vibes):
            color_code = self.get_color_code(vibe["color"])
            pattern = self.get_pattern_for_vibe(vibe)
            reset = self.reset_color
            
            lines.append(f"Step {i+1}: {color_code}{vibe['vibe_type']}{reset} [{vibe['energy_level']}/100]")
            lines.append(f"{color_code}{pattern * 5}{reset}")
            
            # Add arrow except for the last item
            if i < len(vibes) - 1:
                lines.append("      ↓")
                
        lines.append("=" * 40)
        return "\n".join(lines)
    
    def create_vibe_report_visualization(self, report: Dict[str, Any]) -> str:
        """
        Create a visual representation of a vibe report.
        
        Args:
            report: A vibe report dictionary
            
        Returns:
            String visualization of the report
        """
        vibe = report["vibe"]
        color_code = self.get_color_code(vibe["color"])
        reset = self.reset_color
        
        lines = []
        lines.append(f"{'=' * 50}")
        lines.append(f"{color_code}VIBE REPORT - {time.strftime('%Y-%m-%d %H:%M:%S')}{reset}")
        lines.append(f"{'=' * 50}")
        lines.append(f"\n{color_code}{self.visualize_vibe(vibe)}{reset}\n")
        
        lines.append(f"DESCRIPTION:\n{report['description']}\n")
        
        lines.append("RECOMMENDED ACTIVITIES:")
        for i, activity in enumerate(report["recommended_activities"]):
            lines.append(f"  {i+1}. {activity}")
        lines.append("")
        
        lines.append("PLAYLIST:")
        for i, track in enumerate(report["playlist"]):
            lines.append(f"  {i+1}. {track}")
        lines.append("")
        
        lines.append(f"QUOTE: \"{report['inspirational_quote']}\"")
        lines.append(f"{'=' * 50}")
        
        return "\n".join(lines)