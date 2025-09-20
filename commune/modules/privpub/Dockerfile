FROM rust:1.75-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy source files
COPY . .

# Build the application
RUN cargo build --release

# Create a minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binary
COPY --from=0 /app/commune/modules/privpub/target/release/sr25519-hard-derive-attest /usr/local/bin/

# Set the entrypoint
ENTRYPOINT ["sr25519-hard-derive-attest"]
