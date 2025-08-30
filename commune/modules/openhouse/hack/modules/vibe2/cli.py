#!/usr/bin/env python
import argparse
import time
import sys
from typing import Dict, Any, List, Optional

from .vibe import Vibe, create_vibe_playlist, get_vibe_quote
from .vibe_generator import VibeGenerator
from .visualizer import VibeVisualizer

class VibeCLI:
    """
    Command-line interface for the Vibe module.
    """
    
    def __init__(self):
        self.generator = VibeGenerator()
        self.visualizer = VibeVisualizer()
        
    def generate_and_display(self) -> None:
        """
        Generate a new vibe and display it.
        """
        vibe = self.generator.generate_vibe()
        print(self.visualizer.visualize_vibe(vibe))
        print(f"\nDescription: {self.generator.vibe_engine.get_vibe_description(vibe)}")
        
    def generate_report(self) -> None:
        """
        Generate and display a comprehensive vibe report.
        """
        report = self.generator.get_vibe_report()
        print(self.visualizer.create_vibe_report_visualization(report))
    
    def show_evolution(self, steps: int = 5) -> None:
        """
        Show an evolution of vibes.
        
        Args:
            steps: Number of evolution steps
        """
        evolution = self.generator.vibe_evolution(steps)
        print(self.visualizer.visualize_vibe_evolution(evolution))
    
    def create_custom(self, args) -> None:
        """
        Create and display a custom vibe.
        
        Args:
            args: Command line arguments
        """
        vibe = self.generator.create_custom_vibe(
            vibe_type=args.type,
            mood=args.mood,
            color=args.color,
            energy_level=args.energy
        )
        print(self.visualizer.visualize_vibe(vibe))
        
    def show_quote(self) -> None:
        """
        Display a random vibe quote.
        """
        print(f"\n\"{get_vibe_quote()}\"\n")
    
    def show_playlist(self, vibe_type: str) -> None:
        """
        Show a playlist for a specific vibe type.
        
        Args:
            vibe_type: Type of vibe for the playlist
        """
        playlist = create_vibe_playlist(vibe_type)
        print(f"\nPlaylist for {vibe_type} vibe:\n")  
        for i, track in enumerate(playlist):
            print(f"  {i+1}. {track}")
        print()


def main():
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="Generate and manage dope vibes")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    subparsers.add_parser("generate", help="Generate a random vibe")
    
    # Report command
    subparsers.add_parser("report", help="Generate a comprehensive vibe report")
    
    # Evolution command
    evolution_parser = subparsers.add_parser("evolve", help="Show an evolution of vibes")
    evolution_parser.add_argument("-s", "--steps", type=int, default=5, help="Number of evolution steps")
    
    # Custom command
    custom_parser = subparsers.add_parser("custom", help="Create a custom vibe")
    custom_parser.add_argument("-t", "--type", required=True, help="Type of vibe")
    custom_parser.add_argument("-m", "--mood", required=True, help="Mood of the vibe")
    custom_parser.add_argument("-c", "--color", required=True, help="Color associated with the vibe")
    custom_parser.add_argument("-e", "--energy", type=int, required=True, help="Energy level (1-100)")
    
    # Quote command
    subparsers.add_parser("quote", help="Show a random vibe quote")
    
    # Playlist command
    playlist_parser = subparsers.add_parser("playlist", help="Show a playlist for a vibe type")
    playlist_parser.add_argument("-t", "--type", required=True, 
                              choices=["chill", "energetic", "dreamy", "nostalgic", "cosmic"],
                              help="Type of vibe for the playlist")
    
    args = parser.parse_args()
    cli = VibeCLI()
    
    if args.command == "generate":
        cli.generate_and_display()
    elif args.command == "report":
        cli.generate_report()
    elif args.command == "evolve":
        cli.show_evolution(args.steps)
    elif args.command == "custom":
        cli.create_custom(args)
    elif args.command == "quote":
        cli.show_quote()
    elif args.command == "playlist":
        cli.show_playlist(args.type)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()