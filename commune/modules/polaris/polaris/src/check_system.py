#!/usr/bin/env python3
"""
System Information Checker

This script detects your system's hardware information and displays it in a readable format.
It helps diagnose issues with hardware detection for the Polaris Cloud system.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

# Add parent directory to path so we can import system_info
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir.parent.parent))

from polariscloud.src import system_info

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_info(label, value, indent=0):
    """Print a labeled value with optional indentation."""
    indent_str = " " * indent
    print(f"{indent_str}{label}: {value}")

def run_system_check(force_type=None):
    """Run system information detection and print results."""
    print_header("SYSTEM DETECTION STARTED")
    print("Running system information detection...")
    
    # Force CPU or GPU mode if specified
    if force_type:
        print(f"FORCED MODE: {force_type}")
        sys_info = system_info.get_system_info(resource_type=force_type)
    else:
        sys_info = system_info.get_system_info()
    
    # Get the log file path
    log_path = os.path.join("polariscloud", "logs", "raw_system_info.log")
    
    if sys_info:
        print_header("DETECTION RESULTS")
        
        # System overview
        print_info("Location", sys_info.get("location", "Unknown"))
        
        # Resource information
        if "compute_resources" in sys_info and len(sys_info["compute_resources"]) > 0:
            resource = sys_info["compute_resources"][0]
            print_info("Resource Type", resource.get("resource_type", "Unknown"))
            print_info("System RAM", resource.get("ram", "Unknown"))
            
            if "storage" in resource:
                storage = resource["storage"]
                print_info("Storage Type", storage.get("type", "Unknown"))
                print_info("Storage Capacity", storage.get("capacity", "Unknown"))
            
            # CPU or GPU specific information
            if "cpu_specs" in resource:
                cpu = resource["cpu_specs"]
                print_header("CPU INFORMATION")
                print_info("CPU Name", cpu.get("cpu_name", "Unknown"))
                print_info("Vendor", cpu.get("vendor_id", "Unknown"))
                print_info("Architecture", cpu.get("op_modes", "Unknown"))
                print_info("Total Cores/Threads", cpu.get("total_cpus", "Unknown"))
                print_info("Cores per Socket", cpu.get("cores_per_socket", "Unknown"))
                print_info("Threads per Core", cpu.get("threads_per_core", "Unknown"))
                print_info("CPU Max Speed", f"{cpu.get('cpu_max_mhz', 0)} MHz")
            
            if "gpu_specs" in resource:
                gpu = resource["gpu_specs"]
                print_header("GPU INFORMATION")
                print_info("GPU Name", gpu.get("gpu_name", "Unknown"))
                print_info("Memory", gpu.get("memory_size", "Unknown"))
                if gpu.get("clock_speed"):
                    print_info("Clock Speed", gpu.get("clock_speed", "Unknown"))
                if gpu.get("power_consumption"):
                    print_info("Power Consumption", gpu.get("power_consumption", "Unknown"))
    else:
        print("ERROR: Failed to gather system information")
    
    # Print log file information
    print_header("LOG FILE INFORMATION")
    print(f"Raw detection log is available at: {log_path}")
    
    if os.path.exists(log_path):
        log_size = os.path.getsize(log_path)
        print(f"Log file size: {log_size} bytes")
        print("\nTo view the full log file, use the --show-log option")
    else:
        print("WARNING: Log file not found")

def show_log():
    """Show the contents of the raw detection log file."""
    return system_info.show_raw_gpu_log()
    
def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Check system hardware information")
    parser.add_argument("--cpu", action="store_true", help="Force CPU detection mode")
    parser.add_argument("--gpu", action="store_true", help="Force GPU detection mode")
    parser.add_argument("--show-log", action="store_true", help="Show the raw detection log")
    
    args = parser.parse_args()
    
    if args.show_log:
        show_log()
    else:
        # Set force_type based on arguments
        force_type = None
        if args.cpu:
            force_type = "CPU"
        elif args.gpu:
            force_type = "GPU"
            
        run_system_check(force_type)
        print("\nRun with --show-log to see detailed detection information")

if __name__ == "__main__":
    main() 