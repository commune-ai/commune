#!/usr/bin/env python3

import os
import sys
import argparse

def count_files_by_depth(directory, max_depth=None):
    """
    Count files at each directory depth level.
    
    Args:
        directory (str): The root directory to start counting from
        max_depth (int, optional): Maximum depth to traverse
        
    Returns:
        dict: Dictionary with depth levels as keys and file counts as values
    """
    depth_counts = {}
    
    for root, dirs, files in os.walk(directory):
        # Calculate current depth relative to the starting directory
        rel_path = os.path.relpath(root, directory)
        depth = 0 if rel_path == '.' else rel_path.count(os.sep) + 1
        
        # Stop if we've reached max_depth
        if max_depth is not None and depth > max_depth:
            dirs[:] = []  # Clear dirs list to prevent further recursion
            continue
            
        # Count files at this depth
        if depth not in depth_counts:
            depth_counts[depth] = 0
        depth_counts[depth] += len(files)
    
    return depth_counts

def display_results(depth_counts):
    """
    Display the results in a formatted way.
    
    Args:
        depth_counts (dict): Dictionary with depth levels as keys and file counts as values
    """
    total_files = sum(depth_counts.values())
    
    print(f"\nNavigation Depth Analysis:\n{'-' * 25}")
    print(f"Total files: {total_files}\n")
    print("Depth | Files | Percentage")
    print("-" * 30)
    
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"{depth:5d} | {count:5d} | {percentage:6.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Track file counts by directory depth')
    parser.add_argument('directory', nargs='?', default='.', 
                        help='The directory to analyze (default: current directory)')
    parser.add_argument('--max-depth', '-d', type=int, default=None,
                        help='Maximum depth to traverse')
    
    args = parser.parse_args()
    
    try:
        if not os.path.isdir(args.directory):
            print(f"Error: '{args.directory}' is not a valid directory")
            return 1
            
        depth_counts = count_files_by_depth(args.directory, args.max_depth)
        display_results(depth_counts)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
