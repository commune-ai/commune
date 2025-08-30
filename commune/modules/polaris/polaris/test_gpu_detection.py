#!/usr/bin/env python3
"""
Test script to check if the GPU detection is working properly.
This will run the has_gpu() function from system_info.py and show detailed output.
"""

import sys
import os
import logging

# Configure logging to see all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()
logger.info("Starting GPU detection test...")

# Add the current directory and src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger.info(f"Current directory: {current_dir}")
logger.info(f"Python path: {sys.path}")

# Import the has_gpu function - try different import paths
found = False
error_messages = []

try:
    # Try direct import first
    from system_info import has_gpu
    logger.info("Successfully imported has_gpu from system_info")
    found = True
except ImportError as e:
    error_messages.append(f"Direct import failed: {str(e)}")
    
    try:
        # Try from src
        from src.system_info import has_gpu
        logger.info("Successfully imported has_gpu from src.system_info")
        found = True
    except ImportError as e:
        error_messages.append(f"src.system_info import failed: {str(e)}")
        
        # List files in src directory to debug
        if os.path.exists(src_dir):
            logger.info(f"Files in src directory: {os.listdir(src_dir)}")
        else:
            logger.info("src directory not found")
            
        # Try a manual implementation that uses nvidia-smi directly
        logger.info("Using fallback GPU detection implementation")
        
        def has_gpu():
            """Fallback GPU detection function."""
            import subprocess
            try:
                nvidia_output = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                            capture_output=True, text=True, timeout=5)
                
                if nvidia_output.returncode == 0 and nvidia_output.stdout.strip():
                    gpu_names = nvidia_output.stdout.strip().split('\n')
                    logger.info(f"NVIDIA GPUs found: {gpu_names}")
                    return True
                else:
                    logger.info("nvidia-smi completed but returned no GPUs or failed")
                    return False
            except Exception as e:
                logger.error(f"Error in fallback GPU detection: {e}")
                return False
        
        found = True  # We created our own function

if not found:
    logger.error(f"Could not import has_gpu function: {', '.join(error_messages)}")
    sys.exit(1)

# Run the detection function
logger.info("Running has_gpu() function...")
result = has_gpu()
logger.info(f"has_gpu() result: {result}")

# Try to get more GPU info with nvidia-smi directly
try:
    import subprocess
    nvidia_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if nvidia_info.returncode == 0:
        logger.info("NVIDIA-SMI output:")
        for line in nvidia_info.stdout.split('\n'):
            logger.info(f"  {line}")
    else:
        logger.error(f"nvidia-smi error: {nvidia_info.stderr}")
except Exception as e:
    logger.error(f"Error running nvidia-smi: {e}")

logger.info("GPU detection test complete.") 