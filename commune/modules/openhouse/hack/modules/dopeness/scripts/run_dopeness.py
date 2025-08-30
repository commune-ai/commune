#!/usr/bin/env python3

import sys
import os
import random

# Add parent directory to path so we can import the dopeness module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dopeness import Dopeness

def main():
    # ASCII art banner
    banner = """
    ██████╗  ██████╗ ██████╗ ███████╗███╗   ██╗███████╗███████╗███████╗
    ██╔══██╗██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝██╔════╝
    ██║  ██║██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗  ███████╗███████╗
    ██║  ██║██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║╚════██║
    ██████╔╝╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████║███████║
    ╚═════╝  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝
    """
    
    # Random colors for the banner
    colors = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
    ]
    
    reset_color = "\033[0m"
    color = random.choice(colors)
    
    print(f"{color}{banner}{reset_color}")
    print("Welcome to the Dopeness Generator!\n")
    
    try:
        # Get user input for vibe level and chill factor
        vibe_level = int(input("Enter vibe level (1-10): "))
        chill_factor = float(input("Enter chill factor (0.0-1.0): "))
        
        # Create Dopeness instance
        dope = Dopeness(vibe_level=vibe_level, chill_factor=chill_factor)
        
        print(f"\nCreated a Dopeness with vibe level {dope.vibe_level} and chill factor {dope.chill_factor}")
        
        while True:
            print("\nWhat would you like to do?")
            print("1. Generate a vibe phrase")
            print("2. Perform a vibe check")
            print("3. Enhance text with vibes")
            print("4. Generate a vibe playlist")
            print("5. Take a chill moment")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == "1":
                print(f"\nYour vibe: {dope.generate_vibe()}")
            
            elif choice == "2":
                text = input("\nEnter text to vibe check: ")
                result = dope.vibe_check(text)
                status = "PASSED" if result["passed"] else "FAILED"
                print(f"Vibe check {status} with score {result['vibe_score']}")
                print(f"Message: {result['message']}")
            
            elif choice == "3":
                text = input("\nEnter text to enhance: ")
                enhanced = dope.enhance_vibes(text)
                print(f"\nEnhanced text: {enhanced}")
            
            elif choice == "4":
                print("\nMood options: 'chill', 'hype', or leave blank for mixed")
                mood = input("Enter mood: ").lower() or None
                if mood not in ["chill", "hype", None]:
                    mood = None
                
                playlist = dope.vibe_playlist(mood=mood)
                print(f"\nYour {mood or 'mixed'} playlist:")
                for i, track in enumerate(playlist, 1):
                    print(f"{i}. {track}")
            
            elif choice == "5":
                try:
                    duration = float(input("\nHow long to chill (seconds): "))
                    print(dope.chill_out(duration))
                except ValueError:
                    print("Please enter a valid number of seconds.")
            
            elif choice == "6":
                print("\nStay dope! ✌️")
                break
            
            else:
                print("\nInvalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nExiting. Keep those vibes high! ✌️")
    except ValueError:
        print("\nPlease enter valid numbers for vibe level and chill factor.")

if __name__ == "__main__":
    main()