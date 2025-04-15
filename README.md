
# Commune

A containerized application framework with easy management scripts.

## Usage

### Basic Commands

```bash
# Start the application
make start

# Stop the application
make stop

# Enter the container shell
make enter

# Build the container
make build

# Restart the application
make restart

# Run tests
make test

# Install dependencies
make install
```

### Advanced Usage with Arguments

You can also pass specific arguments to the scripts:

```bash
# Start with a custom name
make start-with-name name=myapp

# Enter a specific container
make enter-with-name name=myapp

# Stop a specific container
make stop-with-name name=myapp
```

Alternatively, you can use the scripts directly:

```bash
# Start with various options
./scripts/start.sh --name=myapp --port=8080 --shm-size=8g

# You can also use space after --name
./scripts/start.sh --name myapp

# Enter a specific container
./scripts/enter.sh --name myapp

# Stop a specific container
./scripts/stop.sh --name myapp
```

## Available Options

- `--name=VALUE`: Set the container name
- `--image=VALUE`: Use a specific image
- `--port=VALUE`: Map a specific port
- `--shm-size=VALUE`: Set shared memory size (e.g., 4g)
- `--build`: Build the image before starting
- `--test`: Run tests after starting
