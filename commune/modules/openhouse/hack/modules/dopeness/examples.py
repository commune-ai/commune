from dopeness import Dopeness

def main():
    print("\n🔥✨ DOPENESS EXAMPLES ✨🔥\n")
    
    # Create a standard dopeness instance
    dope = Dopeness(vibe_level=8, chill_factor=0.7)
    print(f"Created Dopeness with vibe level {dope.vibe_level} and chill factor {dope.chill_factor}")
    
    # Generate some vibes
    print("\n--- GENERATING VIBES ---")
    for _ in range(3):
        print(f"Vibe: {dope.generate_vibe()}")
    
    # Vibe check examples
    print("\n--- VIBE CHECKS ---")
    inputs = [
        "Just another boring day",
        "OMG this party is absolutely amazing!",
        "Coding all night long with great music"
    ]
    
    for input_text in inputs:
        result = dope.vibe_check(input_text)
        status = "PASSED" if result["passed"] else "FAILED"
        print(f"Input: '{input_text}'")
        print(f"Result: {status} with score {result['vibe_score']}")
        print(f"Message: {result['message']}\n")
    
    # Text enhancement
    print("\n--- TEXT ENHANCEMENT ---")
    texts = [
        "Meeting at 3pm",
        "Project deadline tomorrow",
        "Weekend plans include beach and movies"
    ]
    
    for text in texts:
        enhanced = dope.enhance_vibes(text)
        print(f"Original: '{text}'")
        print(f"Enhanced: '{enhanced}'\n")
    
    # Playlist generation
    print("\n--- VIBE PLAYLISTS ---")
    print("Chill playlist:")
    for track in dope.vibe_playlist(mood="chill"):
        print(f"- {track}")
    
    print("\nHype playlist:")
    for track in dope.vibe_playlist(mood="hype"):
        print(f"- {track}")
    
    # Chill out
    print("\n--- CHILLING OUT ---")
    print(dope.chill_out(0.2))
    
    print("\n🤙 DOPENESS EXAMPLES COMPLETE 🤙")

if __name__ == "__main__":
    main()