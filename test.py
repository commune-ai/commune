import argparse
import os

def create_directory(directory_name):
    try:
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except OSError as e:
        print(f"Error: {e.strerror}")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Create a new directory.")

    # Add the directory name argument
    parser.add_argument('directory_name', type=str, help='Name of the directory to create')

    # Parse the arguments
    args = parser.parse_args()

    # Access the directory name argument
    directory_name = args.directory_name

    # Create the directory
    create_directory(directory_name)
