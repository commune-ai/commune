 # start of file
#!/bin/bash

# Run benchmarks for the time2stake pallet
cargo run --release -- benchmark pallet \
  --pallet time2stake \
  --extrinsic '*' \
  --steps 50 \
  --repeat 20 \
  --output ./pallets/time2stake/src/weights.rs \
  --template ./.maintain/frame-weight-template.hbs
