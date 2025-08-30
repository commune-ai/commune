#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
import random

class AudioGenerator:
    def __init__(self):
        self.sample_rate = 44100  # Standard audio sample rate
        self.note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'Db': 277.18, 'D': 293.66, 'D#': 311.13,
            'Eb': 311.13, 'E': 329.63, 'F': 349.23, 'F#': 369.99, 'Gb': 369.99,
            'G': 392.00, 'G#': 415.30, 'Ab': 415.30, 'A': 440.00, 'A#': 466.16,
            'Bb': 466.16, 'B': 493.88
        }
        
        # Define scales
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11, 12],
            'minor': [0, 2, 3, 5, 7, 8, 10, 12],
            'pentatonic': [0, 2, 4, 7, 9, 12],
            'blues': [0, 3, 5, 6, 7, 10, 12],
            'dorian': [0, 2, 3, 5, 7, 9, 10, 12]
        }
    
    def generate_note(self, frequency, duration, amplitude=0.5, attack=0.1, release=0.1):
        """Generate a single note with ADSR envelope"""
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate basic sine wave
        note = amplitude * np.sin(frequency * 2 * np.pi * t)
        
        # Apply attack and release (simple ADSR envelope)
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # Attack phase
        if attack_samples > 0:
            attack_env = np.linspace(0, 1, attack_samples)
            note[:attack_samples] *= attack_env
        
        # Release phase
        if release_samples > 0 and num_samples > release_samples:
            release_env = np.linspace(1, 0, release_samples)
            note[-release_samples:] *= release_env
        
        return note
    
    def generate_chord(self, root_note, chord_type='major', duration=1.0, amplitude=0.3):
        """Generate a chord based on root note and chord type"""
        if root_note not in self.note_frequencies:
            raise ValueError(f"Unknown note: {root_note}")
        
        root_freq = self.note_frequencies[root_note]
        
        # Define chord intervals
        intervals = {
            'major': [0, 4, 7],  # Root, major third, perfect fifth
            'minor': [0, 3, 7],  # Root, minor third, perfect fifth
            'diminished': [0, 3, 6],  # Root, minor third, diminished fifth
            'augmented': [0, 4, 8],  # Root, major third, augmented fifth
            'sus2': [0, 2, 7],  # Root, major second, perfect fifth
            'sus4': [0, 5, 7],  # Root, perfect fourth, perfect fifth
            '7': [0, 4, 7, 10],  # Dominant seventh
            'maj7': [0, 4, 7, 11],  # Major seventh
            'min7': [0, 3, 7, 10]  # Minor seventh
        }
        
        if chord_type not in intervals:
            chord_type = 'major'  # Default to major if invalid
        
        # Generate each note in the chord
        chord = np.zeros(int(self.sample_rate * duration))
        
        for semitones in intervals[chord_type]:
            # Calculate frequency using equal temperament formula
            freq = root_freq * (2 ** (semitones / 12))
            note = self.generate_note(freq, duration, amplitude / len(intervals[chord_type]))
            chord += note
        
        return chord
    
    def generate_melody(self, key, scale_type, num_notes=8, bpm=120):
        """Generate a melody in the given key and scale"""
        if key not in self.note_frequencies:
            raise ValueError(f"Unknown key: {key}")
        
        if scale_type not in self.scales:
            scale_type = 'major'  # Default to major scale
        
        # Calculate note duration based on BPM (quarter notes)
        note_duration = 60 / bpm
        
        # Get the scale degrees
        scale_degrees = self.scales[scale_type]
        
        # Generate the melody
        melody = np.array([])
        
        # Keep track of previous note to avoid large jumps
        prev_degree_idx = random.randint(0, len(scale_degrees) - 1)
        
        for _ in range(num_notes):
            # Choose next note (favor small intervals)
            max_jump = 2  # Maximum jump in scale degrees
            degree_idx = max(0, min(len(scale_degrees) - 1, 
                                    prev_degree_idx + random.randint(-max_jump, max_jump)))
            
            # Get the semitone offset for this scale degree
            semitones = scale_degrees[degree_idx]
            
            # Calculate the frequency
            freq = self.note_frequencies[key] * (2 ** (semitones / 12))
            
            # Randomly vary the duration for more interesting rhythms
            duration_factor = random.choice([0.5, 0.5, 1.0, 1.0, 1.0, 1.5, 2.0])
            duration = note_duration * duration_factor
            
            # Generate the note
            note = self.generate_note(freq, duration, 
                                     amplitude=random.uniform(0.3, 0.7),
                                     attack=random.uniform(0.05, 0.2),
                                     release=random.uniform(0.1, 0.3))
            
            # Add to melody
            melody = np.append(melody, note)
            
            # Update previous degree
            prev_degree_idx = degree_idx
        
        return melody
    
    def apply_effect(self, audio, effect_type, **params):
        """Apply audio effects to the signal"""
        if effect_type == 'reverb':
            # Simple reverb simulation
            delay_ms = params.get('delay_ms', 100)
            decay = params.get('decay', 0.6)
            num_echoes = params.get('num_echoes', 5)
            
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            result = np.copy(audio)
            
            for i in range(1, num_echoes + 1):
                echo = np.zeros_like(audio)
                if delay_samples * i < len(audio):
                    echo[delay_samples * i:] = audio[:len(audio) - delay_samples * i] * (decay ** i)
                    result += echo
            
            # Normalize to avoid clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
                
            return result
            
        elif effect_type == 'delay':
            # Echo effect
            delay_ms = params.get('delay_ms', 300)
            feedback = params.get('feedback', 0.4)
            mix = params.get('mix', 0.5)
            
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            result = np.copy(audio)
            echo = np.zeros_like(audio)
            
            if delay_samples < len(audio):
                echo[delay_samples:] = audio[:len(audio) - delay_samples] * feedback
                result = audio * (1 - mix) + echo * mix
                
            return result
            
        elif effect_type == 'distortion':
            # Simple distortion
            gain = params.get('gain', 10)
            mix = params.get('mix', 0.5)
            
            # Apply gain and clip
            distorted = np.clip(audio * gain, -1.0, 1.0)
            
            # Mix with original
            result = audio * (1 - mix) + distorted * mix
            return result
            
        elif effect_type == 'chorus':
            # Simple chorus effect
            rate = params.get('rate', 0.5)  # Hz
            depth = params.get('depth', 0.003)  # seconds
            mix = params.get('mix', 0.5)
            
            # Create LFO for chorus
            t = np.arange(len(audio)) / self.sample_rate
            lfo = depth * self.sample_rate * np.sin(2 * np.pi * rate * t)
            
            # Apply chorus effect
            chorus = np.zeros_like(audio)
            for i in range(len(audio)):
                idx = i + int(lfo[i])
                if 0 <= idx < len(audio):
                    chorus[i] = audio[idx]
            
            # Mix with original
            result = audio * (1 - mix) + chorus * mix
            return result
            
        else:
            # No effect or unknown effect
            return audio
    
    def save_audio(self, audio_data, filename='output.wav'):
        """Save audio data to a WAV file"""
        # Ensure the audio is normalized
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Convert to 16-bit PCM
        audio_data_16bit = (audio_data * 32767).astype(np.int16)
        
        # Save to file
        wavfile.write(filename, self.sample_rate, audio_data_16bit)
        return filename
    
    def generate_vibe(self, vibe_params):
        """Generate a complete audio vibe based on parameters"""
        # Extract parameters
        key = vibe_params.get('key', 'C')
        scale = vibe_params.get('scale', 'major')
        bpm = vibe_params.get('bpm', 120)
        instruments = vibe_params.get('instruments', ['synth'])
        effects = vibe_params.get('effects', [])
        intensity = vibe_params.get('intensity', 0.7)
        
        # Calculate durations
        beat_duration = 60 / bpm
        bar_duration = 4 * beat_duration  # 4/4 time signature
        
        # Generate 4 bars of music
        num_bars = 4
        total_duration = num_bars * bar_duration
        
        # Initialize the audio array
        total_samples = int(self.sample_rate * total_duration)
        audio = np.zeros(total_samples)
        
        # Generate chord progression (simple I-IV-V-I)
        chord_progression = ['I', 'IV', 'V', 'I']
        
        # Map chord numerals to semitone offsets in the key
        chord_map = {
            'I': 0,   # Tonic
            'ii': 2,  # Supertonic
            'iii': 4, # Mediant
            'IV': 5,  # Subdominant
            'V': 7,   # Dominant
            'vi': 9,  # Submediant
            'vii': 11 # Leading tone
        }
        
        # Determine chord types based on scale
        if scale == 'major':
            chord_types = {
                'I': 'major', 'ii': 'minor', 'iii': 'minor', 'IV': 'major',
                'V': 'major', 'vi': 'minor', 'vii': 'diminished'
            }
        else:  # Assume minor
            chord_types = {
                'I': 'minor', 'ii': 'diminished', 'III': 'major', 'iv': 'minor',
                'v': 'minor', 'VI': 'major', 'VII': 'major'
            }
        
        # Generate chords
        for i, numeral in enumerate(chord_progression):
            # Get the root note for this chord
            semitones = chord_map.get(numeral, 0)
            root_freq = self.note_frequencies[key] * (2 ** (semitones / 12))
            
            # Find the closest named note
            root_note = key
            min_diff = float('inf')
            for note, freq in self.note_frequencies.items():
                if abs(freq - root_freq) < min_diff:
                    min_diff = abs(freq - root_freq)
                    root_note = note
            
            # Get chord type
            chord_type = chord_types.get(numeral, 'major')
            
            # Generate the chord
            chord = self.generate_chord(root_note, chord_type, bar_duration, amplitude=0.4 * intensity)
            
            # Add to the audio at the right position
            start_sample = i * int(bar_duration * self.sample_rate)
            end_sample = start_sample + len(chord)
            if end_sample <= total_samples:
                audio[start_sample:end_sample] += chord
        
        # Generate melody
        if 'synth' in instruments or 'piano' in instruments:
            melody = self.generate_melody(key, scale, num_notes=16, bpm=bpm)
            
            # Ensure melody fits within the total duration
            if len(melody) > total_samples:
                melody = melody[:total_samples]
            
            # Add melody to audio
            audio[:len(melody)] += melody * 0.5 * intensity
        
        # Add bass line if 'bass' in instruments
        if 'bass' in instruments:
            bass_line = np.array([])
            
            for numeral in chord_progression:
                # Get the root note for this chord
                semitones = chord_map.get(numeral, 0)
                
                # Bass plays the root note an octave lower
                freq = self.note_frequencies[key] * (2 ** ((semitones - 12) / 12))
                
                # Generate bass note for a whole bar
                bass_note = self.generate_note(freq, bar_duration, 
                                             amplitude=0.6 * intensity,
                                             attack=0.1, release=0.2)
                
                bass_line = np.append(bass_line, bass_note)
            
            # Ensure bass fits within the total duration
            if len(bass_line) > total_samples:
                bass_line = bass_line[:total_samples]
            
            # Add bass to audio
            audio[:len(bass_line)] += bass_line
        
        # Add drums if 'drums' in instruments
        if 'drums' in instruments:
            # Simple drum pattern
            kick_pattern = [1, 0, 0, 0, 1, 0, 0, 0]  # Kick on 1 and 3
            snare_pattern = [0, 0, 1, 0, 0, 0, 1, 0]  # Snare on 2 and 4
            hihat_pattern = [1, 1, 1, 1, 1, 1, 1, 1]  # Hihat on every beat
            
            # Generate one bar of drums
            drum_bar = np.zeros(int(bar_duration * self.sample_rate))
            beats_per_bar = 8  # 8 eighth notes per bar
            beat_samples = int(bar_duration * self.sample_rate / beats_per_bar)
            
            for i in range(beats_per_bar):
                # Kick drum (low frequency sine with quick decay)
                if kick_pattern[i]:
                    kick = self.generate_note(60, 0.1, amplitude=0.8 * intensity, attack=0.01, release=0.1)
                    start = i * beat_samples
                    end = start + len(kick)
                    if end <= len(drum_bar):
                        drum_bar[start:end] += kick
                
                # Snare (noise with bandpass filter - simplified here as noise)
                if snare_pattern[i]:
                    snare_duration = 0.1
                    snare_samples = int(snare_duration * self.sample_rate)
                    snare = np.random.uniform(-0.5, 0.5, snare_samples) * 0.7 * intensity
                    # Apply simple envelope
                    env = np.exp(-np.linspace(0, 5, snare_samples))
                    snare = snare * env
                    
                    start = i * beat_samples
                    end = start + len(snare)
                    if end <= len(drum_bar):
                        drum_bar[start:end] += snare
                
                # Hi-hat (high frequency noise with very quick decay)
                if hihat_pattern[i]:
                    hihat_duration = 0.05
                    hihat_samples = int(hihat_duration * self.sample_rate)
                    hihat = np.random.uniform(-0.3, 0.3, hihat_samples) * 0.4 * intensity
                    # Apply quick decay
                    env = np.exp(-np.linspace(0, 10, hihat_samples))
                    hihat = hihat * env
                    
                    start = i * beat_samples
                    end = start + len(hihat)
                    if end <= len(drum_bar):
                        drum_bar[start:end] += hihat
            
            # Repeat drum pattern for all bars
            for i in range(num_bars):
                start = i * len(drum_bar)
                end = start + len(drum_bar)
                if end <= total_samples:
                    audio[start:end] += drum_bar
        
        # Apply effects
        for effect in effects:
            if effect == 'reverb':
                audio = self.apply_effect(audio, 'reverb', 
                                         delay_ms=100, decay=0.5, num_echoes=5)
            elif effect == 'delay':
                audio = self.apply_effect(audio, 'delay', 
                                         delay_ms=int(beat_duration * 1000 / 2), 
                                         feedback=0.4, mix=0.3)
            elif effect == 'distortion':
                audio = self.apply_effect(audio, 'distortion', 
                                         gain=3 + 5 * intensity, mix=0.3)
            elif effect == 'chorus':
                audio = self.apply_effect(audio, 'chorus', 
                                         rate=0.5, depth=0.003, mix=0.3)
        
        return audio

# Example usage
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # Create an instance of the audio generator
    generator = AudioGenerator()
    
    # Generate a vibe
    vibe_params = {
        'key': 'C',
        'scale': 'minor',
        'bpm': 100,
        'instruments': ['synth', 'bass', 'drums'],
        'effects': ['reverb', 'delay'],
        'intensity': 0.8
    }
    
    print("Generating vibe...")
    audio = generator.generate_vibe(vibe_params)
    
    # Save the audio
    output_file = generator.save_audio(audio, 'output/vibe_output.wav')
    print(f"Saved audio to {output_file}")
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / generator.sample_rate, len(audio)), audio)
    plt.title('Generated Vibe Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('output/waveform.png')
    plt.close()
    
    print("Done!")
