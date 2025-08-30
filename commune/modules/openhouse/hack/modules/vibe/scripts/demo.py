#!/usr/bin/env python3

import sys
import os
import random
import time
from colorama import init, Fore, Style

# Add the parent directory to the path so we can import the vibe module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibe import Vibe

def print_color(text, color):
    """Print text in a specified color"""
    colors = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE
    }
    print(f"{colors.get(color, Fore.WHITE)}{text}{Style.RESET_ALL}")

def print_with_typing_effect(text, delay=0.03):
    """Print text with a typing effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def display_vibe(vibe_data):
    """Display vibe information in a visually appealing way"""
    vibe_type = vibe_data["vibe_type"]
    quote = vibe_data["quote"]
    colors = vibe_data["colors"]
    music = vibe_data["music"]
    
    # Map vibe types to display colors
    vibe_colors = {
        "chill": "blue",
        "energetic": "red",
        "creative": "magenta",
        "focused": "cyan",
        "happy": "yellow",
        "reflective": "blue",
        "motivated": "red",
        "peaceful": "green"
    }
    
    display_color = vibe_colors.get(vibe_type, "white")
    
    # Header
    print("\n" + "="*60)
    print_color(f"  🌟  {vibe_type.upper()} VIBE  🌟", display_color)
    print("="*60)
    
    # Quote
    print("\n📝 INSPIRATIONAL QUOTE:")
    print_with_typing_effect(f'  "{quote}"')
    
    # Music
    print("\n🎵 MUSIC RECOMMENDATION:")
    print_color(f"  {music['artist']} - {music['song']}", display_color)
    print(f"  Link: {music['link']}")
    
    # Colors
    print("\n🎨 COLOR PALETTE:")
    for color in colors:
        print_color(f"  {color}", display_color)
    
    print("\n" + "="*60)
    print_color("  Enjoy your dope vibe! ✨", display_color)
    print("="*60 + "\n")

def main():
    """Main function to demonstrate the vibe module"""
    # Initialize colorama for cross-platform color support
    init()
    
    # Create vibe instance
    vibe = Vibe()
    
    # Get all available vibe types
    vibe_types = vibe.vibe_types
    
    # Welcome message
    print("\n" + "="*60)
    print_color("  🌈  WELCOME TO THE DOPE VIBE GENERATOR  🌈", "cyan")
    print("="*60)
    
    print("\nThis tool helps you generate positive vibes through quotes,")
    print("color schemes, and music recommendations.")
    
    # Show available vibe types
    print("\nAvailable vibe types:")
    for i, vibe_type in enumerate(vibe_types, 1):
        print_color(f"  {i}. {vibe_type}", "yellow")
    
    print("\nOptions:")
    print_color("  0. Random vibe", "green")
    print_color("  1-8. Select a specific vibe", "green")
    print_color("  q. Quit", "red")
    
    while True:
        choice = input("\nEnter your choice: ")
        
        if choice.lower() == 'q':
            print_color("\nThanks for using the Dope Vibe Generator! Stay positive! ✌️\n", "cyan")
            break
        
        try:
            if choice == '0':
                # Random vibe
                vibe_data = vibe.get_vibe()
                display_vibe(vibe_data)
            elif 1 <= int(choice) <= len(vibe_types):
                # Specific vibe
                selected_vibe = vibe_types[int(choice) - 1]
                vibe_data = vibe.get_vibe(vibe_type=selected_vibe)
                display_vibe(vibe_data)
            else:
                print_color("Invalid choice. Please try again.", "red")
        except ValueError:
            print_color("Invalid input. Please enter a number or 'q' to quit.", "red")

if __name__ == "__main__":
    main()