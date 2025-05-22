 # start of file
#!/usr/bin/env python3
"""
Docker container monitoring script.
Similar to PM2's monitoring interface but for Docker containers.
"""

import commune as c
import time
import argparse
import os
import sys

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_size(size_str):
    """Format size strings for consistent display."""
    if not size_str:
        return "0B"
    return size_str

def format_percent(percent_str):
    """Format percentage strings for consistent display."""
    if not percent_str:
        return "0.00%"
    if not percent_str.endswith('%'):
        percent_str += '%'
    return percent_str

def main():
    parser = argparse.ArgumentParser(description='Monitor Docker containers')
    parser.add_argument('--refresh', type=int, default=2, help='Refresh interval in seconds')
    parser.add_argument('--container', type=str, help='Monitor specific container')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear screen between updates')
    args = parser.parse_args()

    docker = c.module('docker')()
    
    try:
        while True:
            if not args.no_clear:
                clear_screen()
            
            print("\033[1m" + "Docker Container Monitor" + "\033[0m")
            print("\033[1m" + "─" * 100 + "\033[0m")
            
            # Get container list
            containers = docker.list(all=True)
            if containers.empty:
                print("No containers found.")
                time.sleep(args.refresh)
                continue
                
            # Get container stats
            stats = docker.cstats(update=True)
            
            # Filter for specific container if requested
            if args.container:
                containers = containers[containers['names'].str.contains(args.container)]
                if not containers.empty and not stats.empty:
                    stats = stats[stats['name'].str.contains(args.container)]
            
            # Print container information
            print("\033[1m{:<20} {:<15} {:<10} {:<10} {:<15} {:<15} {:<15}\033[0m".format(
                "Container", "Status", "CPU %", "MEM %", "MEM Usage", "NET I/O", "BLOCK I/O"
            ))
            print("─" * 100)
            
            for _, container in containers.iterrows():
                name = container.get('names', '')
                status = container.get('status', '')
                
                # Find matching stats
                container_stats = stats[stats['name'] == name] if not stats.empty else None
                
                if container_stats is not None and not container_stats.empty:
                    row = container_stats.iloc[0]
                    cpu = format_percent(row.get('cpu%', '0%'))
                    mem = format_percent(row.get('mem%', '0%'))
                    mem_usage = format_size(row.get('mem_usage', '0B'))
                    net_io = f"{format_size(row.get('net_in', '0B'))}/{format_size(row.get('net_out', '0B'))}"
                    block_io = f"{format_size(row.get('block_in', '0B'))}/{format_size(row.get('block_out', '0B'))}"
                else:
                    cpu = "N/A"
                    mem = "N/A"
                    mem_usage = "N/A"
                    net_io = "N/A"
                    block_io = "N/A"
                
                # Color status
                if 'Up' in status:
                    status = f"\033[92m{status}\033[0m"  # Green for running
                elif 'Exited' in status:
                    status = f"\033[91m{status}\033[0m"  # Red for stopped
                else:
                    status = f"\033[93m{status}\033[0m"  # Yellow for other states
                
                print("{:<20} {:<40} {:<10} {:<10} {:<15} {:<15} {:<15}".format(
                    name, status, cpu, mem, mem_usage, net_io, block_io
                ))
            
            print("\n" + "─" * 100)
            print(f"Refresh: every {args.refresh}s | Press Ctrl+C to exit")
            
            time.sleep(args.refresh)
            
    except KeyboardInterrupt:
        print("\nExiting Docker monitor...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
