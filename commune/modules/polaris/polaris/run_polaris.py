#!/usr/bin/env python3
"""
Direct runner for Polaris CLI.
This script directly imports and runs the CLI without relying on entry points.
"""

import sys
from polaris_cli.cli import cli

if __name__ == "__main__":
    # Remove the script name from arguments
    args = sys.argv[1:]
    
    # Pass the remaining arguments to the CLI
    cli(args) 