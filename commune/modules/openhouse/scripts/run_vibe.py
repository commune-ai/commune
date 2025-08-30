#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audio_generator import AudioGenerator
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Generate a dope vibe audio file')
    parser.add_argument('--key', type=str, default='C', help='Musical key (e.g., C, F#, Bb)')
    parser.add_argument('--scale', type=str, default='minor', 
                        choices=['major', 'minor', 'pentatonic', 'blues', 'dorian'],
                        help='Musical scale')
    parser.add_argument('--bpm', type=int, default=100, help='Beats per minute')
    parser.add_argument('--instruments', type=str, default='synth,bass,drums',
                       help='Comma-separated list of instruments')
    parser.add_argument('--effects', type=str, default='reverb,delay',
                       help='Comma-separated list of effects')
    parser.add_argument('--intensity', type=float, default=0.8,
                       help='Intensity of the vibe (0.1 to 1.0)')
    parser.add_argument('--output', type=str, default='output/vibe_output.wav',
                       help='Output file path')
    parser.add_argument('--preset', type=str, default=None,
                       choices=['chill', 'energetic', 'dreamy'],
                       help='Use a preset vibe configuration')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the audio generator
    generator = AudioGenerator()
    
    # Set up vibe parameters
    if args.preset:
        # Preset configurations
        presets = {
            'chill': {
                'key': 'A',
                'scale': 'minor',
                'bpm': 80,
                'instruments': ['piano', 'synth', 'bass'],
                'effects': ['reverb', 'delay'],
                'intensity': 0.7
            },
            'energetic': {
                'key': 'E',
                'scale': 'major',
                'bpm': 130,
                'instruments': ['drums', 'synth', 'electric guitar'],
                'effects': ['distortion', 'compression'],
                'intensity': 0.9
            },
            'dreamy': {
                'key': 'F',
                'scale': 'dorian',
                'bpm': 70,
                'instruments': ['pad', 'bells', 'strings'],
                'effects': ['reverb', 'chorus', 'phaser'],
                'intensity': 0.6
            }
        }
        vibe_params = presets[args.preset]
    else:
        # Custom configuration from command line arguments
        vibe_params = {
            'key': args.key,
            'scale': args.scale,
            'bpm': args.bpm,
            'instruments': args.instruments.split(','),
            'effects': args.effects.split(','),
            'intensity': args.intensity
        }
    
    print(f"Generating vibe with parameters: {json.dumps(vibe_params, indent=2)}")
    
    # Generate the vibe
    audio = generator.generate_vibe(vibe_params)
    
    # Save the audio
    output_file = generator.save_audio(audio, args.output)
    print(f"Saved audio to {output_file}")
    print("Done!")

if __name__ == "__main__":
    main()
