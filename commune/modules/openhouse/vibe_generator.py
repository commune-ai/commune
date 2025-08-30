#!/usr/bin/env python3

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
import random
import time

app = Flask(__name__)

class VibeGenerator:
    def __init__(self):
        self.vibe_types = {
            'chill': {
                'bpm': (70, 90),
                'key': ['C', 'G', 'A'],
                'scale': ['minor', 'major'],
                'instruments': ['piano', 'synth', 'bass'],
                'effects': ['reverb', 'delay'],
                'color_scheme': ['blue', 'purple', 'teal']
            },
            'energetic': {
                'bpm': (120, 140),
                'key': ['D', 'E', 'A'],
                'scale': ['major', 'pentatonic'],
                'instruments': ['drums', 'synth', 'electric guitar'],
                'effects': ['distortion', 'compression'],
                'color_scheme': ['red', 'orange', 'yellow']
            },
            'dreamy': {
                'bpm': (60, 80),
                'key': ['F', 'Bb', 'Eb'],
                'scale': ['minor', 'dorian'],
                'instruments': ['pad', 'bells', 'strings'],
                'effects': ['reverb', 'chorus', 'phaser'],
                'color_scheme': ['lavender', 'pink', 'light blue']
            }
        }
        self.current_vibe = None
        
    def generate_vibe(self, vibe_type='random', intensity=0.7):
        if vibe_type == 'random':
            vibe_type = random.choice(list(self.vibe_types.keys()))
        
        if vibe_type not in self.vibe_types:
            vibe_type = 'chill'  # default to chill if invalid type
            
        vibe_params = self.vibe_types[vibe_type]
        
        # Generate specific parameters based on the vibe type
        bpm = random.randint(*vibe_params['bpm'])
        key = random.choice(vibe_params['key'])
        scale = random.choice(vibe_params['scale'])
        instruments = random.sample(vibe_params['instruments'], 
                                   k=min(3, len(vibe_params['instruments'])))
        effects = random.sample(vibe_params['effects'], 
                               k=min(2, len(vibe_params['effects'])))
        colors = random.sample(vibe_params['color_scheme'], 
                              k=min(2, len(vibe_params['color_scheme'])))
        
        # Create the vibe object
        self.current_vibe = {
            'type': vibe_type,
            'bpm': bpm,
            'key': key,
            'scale': scale,
            'instruments': instruments,
            'effects': effects,
            'colors': colors,
            'intensity': intensity,
            'timestamp': time.time(),
            'id': f"vibe_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        return self.current_vibe
    
    def modify_vibe(self, parameter, value):
        """Modify a specific parameter of the current vibe"""
        if not self.current_vibe:
            return None
            
        if parameter in self.current_vibe:
            self.current_vibe[parameter] = value
            self.current_vibe['timestamp'] = time.time()
            
        return self.current_vibe

# Initialize the vibe generator
vibe_generator = VibeGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_vibe_api():
    data = request.json
    vibe_type = data.get('type', 'random')
    intensity = float(data.get('intensity', 0.7))
    
    vibe = vibe_generator.generate_vibe(vibe_type, intensity)
    return jsonify(vibe)

@app.route('/api/modify', methods=['POST'])
def modify_vibe_api():
    data = request.json
    parameter = data.get('parameter')
    value = data.get('value')
    
    if not parameter or value is None:
        return jsonify({'error': 'Missing parameter or value'}), 400
        
    vibe = vibe_generator.modify_vibe(parameter, value)
    if not vibe:
        return jsonify({'error': 'No active vibe to modify'}), 404
        
    return jsonify(vibe)

@app.route('/api/current')
def get_current_vibe():
    if not vibe_generator.current_vibe:
        vibe_generator.generate_vibe()
    return jsonify(vibe_generator.current_vibe)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        
    app.run(debug=True)
