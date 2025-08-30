#!/bin/bash

# Simple script to run the Vibe Generator

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Navigate to the project directory
cd "$PROJECT_DIR"

# Check if the first argument is provided
if [ -z "$1" ]; then
    # No argument, run in web mode
    echo "Starting Vibe Generator in web mode..."
    python main.py --mode web
else
    # Argument provided, run in CLI mode with the specified vibe
    echo "Starting Vibe Generator with '$1' vibe..."
    python main.py --mode cli --vibe "$1" --duration 300
fi
