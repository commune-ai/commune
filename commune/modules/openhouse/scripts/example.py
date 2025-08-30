import sys
import os

# Add parent directory to path so we can import hack_app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hack_app import HackApp

def main():
    print("Running example script for HackApp")
    
    # Create app instance
    app = HackApp()
    
    # Process multiple data items
    test_data = [
        "Hello, world!",
        {"key": "value"},
        [1, 2, 3, 4, 5],
        42
    ]
    
    for data in test_data:
        app.process_data(data)
    
    # Display all results
    app.display_results()
    
    # You can also get the results programmatically
    results = app.get_results()
    print(f"\nRetrieved {len(results)} results programmatically")

if __name__ == "__main__":
    main()
