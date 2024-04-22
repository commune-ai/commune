import subprocess
from printrun.printcore import printcore
from printrun import gcoder
import time
import os

# Configuration for PrusaSlicer
PRUSASLICER_PATH = "your-path-to-prusa-slicer-console.exe"  # Update with the actual path
SLICER_CONFIG_PATH = "your-path-to-PrusaSlicerConfig.ini" # Update with the actual path
OUTPUT_GCODE_PATH = "your-path-to-AlmechE/output.gcode" # Update with the actual path

# Configuration for printer
PRINTER_PORT = 'COM5'  # Adjust to your printer's serial port
PRINTER_BAUD = 115200  # Adjust to your printer's baud rate

def find_latest_stl(base_dir):
    # The base directory where the folders are saved. Adjust if your base path is different.
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"The specified base directory does not exist: {base_dir}")

    # Get all subdirectories in the base directory that start with 'output_'
    subdirectories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('output_')]
    
    if not subdirectories:
        raise FileNotFoundError("No output directories found.")

    # Find the most recent subdirectory
    latest_dir = max(subdirectories, key=os.path.getmtime)

    # Get all STL files in the latest directory
    stl_files = [f for f in os.listdir(latest_dir) if f.endswith('.stl')]
    
    if not stl_files:
        raise FileNotFoundError("No STL files found in the latest directory.")
    
    # Find the most recent STL file
    latest_stl = max(stl_files, key=lambda f: os.path.getctime(os.path.join(latest_dir, f)))
    return os.path.join(latest_dir, latest_stl)

def slice_with_prusaslicer(stl_path):
    # Scale factor: 1000 (to convert meters to millimeters)
    scale_factor = "1000"

    command = [
        PRUSASLICER_PATH,
        "--load", SLICER_CONFIG_PATH,
        "--scale", scale_factor,  # Add the scale factor
        "--export-gcode",
        "--output", OUTPUT_GCODE_PATH,
        stl_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Standard Output: {result.stdout.decode()}")
    print(f"Standard Error: {result.stderr.decode()}")
    if result.returncode != 0:
        print("Slicing failed.")
        exit(1)

def send_gcode_to_printer():
    p = printcore(PRINTER_PORT, PRINTER_BAUD)
    while not p.online:
        time.sleep(0.1)
    gcode = [line.strip() for line in open(OUTPUT_GCODE_PATH)]
    gcode = gcoder.LightGCode(gcode)
    p.startprint(gcode)
    while p.printing:
        time.sleep(1)
    p.disconnect()

def main():
    try:
        base_dir = "C:/Users/Guest1/Desktop/AutoInvent"  # Update this path to the correct one
        latest_stl_path = find_latest_stl(base_dir)  # Pass base_dir as an argument
        print(f"Latest STL file found: {latest_stl_path}")

        slice_with_prusaslicer(latest_stl_path)
        send_gcode_to_printer()
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
