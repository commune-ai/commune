#!/usr/bin/env python3

"""
A simple script to say hello and provide system information.
"""

import platform
import datetime
import os


def main():
    """Display a friendly greeting and system information."""
    print("\n" + "="*50)
    print("HELLO FROM MR. ROBOT!")
    print("="*50)
    
    # Current time
    now = datetime.datetime.now()
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System info
    print(f"\nSystem Information:")
    print(f"  - OS: {platform.system()} {platform.release()}")
    print(f"  - Python: {platform.python_version()}")
    print(f"  - Machine: {platform.machine()}")
    
    # Current directory
    print(f"\nCurrent directory: {os.getcwd()}")
    
    print("\nReady to assist with your coding tasks!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
