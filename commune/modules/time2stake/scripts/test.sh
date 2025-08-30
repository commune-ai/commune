 # start of file
#!/bin/bash
set -e

# Run the test suite
cargo test --release

echo "Tests completed successfully!"
