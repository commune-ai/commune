import argparse
from vibe_generator import VibeGenerator
from web_interface import create_vibe_interface

def main():
    parser = argparse.ArgumentParser(description="Dope Vibe Generator")
    parser.add_argument(
        "--mode", 
        choices=["cli", "web"], 
        default="web",
        help="Run in CLI mode or web interface mode"
    )
    parser.add_argument(
        "--vibe", 
        type=str,
        help="Start with a specific vibe (only in CLI mode)"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=60,
        help="Duration in seconds to run the vibe (only in CLI mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "web":
        # Start the web interface
        interface = create_vibe_interface()
        interface.launch()
    else:
        # CLI mode
        vibe_gen = VibeGenerator()
        available_vibes = vibe_gen.get_available_vibes()
        
        print("🌈✨ Dope Vibe Generator - CLI Mode ✨🌈")
        print(f"Available vibes: {', '.join(available_vibes)}")
        
        if args.vibe:
            if args.vibe in available_vibes:
                print(f"Starting '{args.vibe}' vibe for {args.duration} seconds...")
                vibe_gen.start_vibe(args.vibe)
                
                try:
                    import time
                    time.sleep(args.duration)
                except KeyboardInterrupt:
                    print("\nVibe interrupted by user.")
                finally:
                    vibe_gen.stop_vibe()
            else:
                print(f"Error: '{args.vibe}' is not a valid vibe.")
                print(f"Choose from: {', '.join(available_vibes)}")
        else:
            # Interactive mode
            print("\nNo vibe specified. Enter a vibe name or 'exit' to quit:")
            while True:
                choice = input("> ").strip().lower()
                if choice == "exit":
                    break
                elif choice == "list":
                    print(f"Available vibes: {', '.join(available_vibes)}")
                elif choice == "stop":
                    vibe_gen.stop_vibe()
                    print("Vibe stopped.")
                elif choice in available_vibes:
                    vibe_gen.start_vibe(choice)
                    print(f"Started '{choice}' vibe. Type 'stop' to end or 'exit' to quit.")
                else:
                    print(f"Unknown vibe: '{choice}'. Type 'list' to see available vibes.")

if __name__ == "__main__":
    main()
