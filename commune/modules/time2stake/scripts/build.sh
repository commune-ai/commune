 # start of file
#!/bin/bash
set -e

# Build the node in release mode
cargo build --release

# Print success message
echo "Build completed successfully!"
echo "You can run the node with: ./target/release/node-template --dev"
