# Uniswap Trader MCP Makefile

.PHONY: install start build-docker run-docker clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make start      - Start the MCP server"
	@echo "  make build-docker - Build Docker image"
	@echo "  make run-docker - Run Docker container"
	@echo "  make clean      - Remove node_modules and logs"

# Install dependencies
install:
	@echo "Installing dependencies..."
	npm install express body-parser @modelcontextprotocol/sdk @uniswap/smart-order-router @uniswap/sdk-core ethers dotenv

# Start the server
start:
	@echo "Starting Uniswap Trader MCP server..."
	./run.sh

# Build Docker image
build-docker:
	@echo "Building Docker image..."
	docker build -t uniswap-trader-mcp .

# Run Docker container
run-docker:
	@echo "Running Docker container..."
	docker run -it --env-file .env uniswap-trader-mcp

# Clean project
clean:
	@echo "Cleaning project..."
	rm -rf node_modules
	rm -f *.log
