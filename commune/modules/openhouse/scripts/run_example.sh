#!/bin/bash

# Run the example script

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the example
python example.py
