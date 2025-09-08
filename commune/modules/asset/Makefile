.PHONY: up down logs ps clean restart

# Start all services
up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services started!"
	@echo "App available at http://localhost:3000"
	@echo "Ethereum RPC at http://localhost:8545"
	@echo "Solana RPC at http://localhost:8899"

# Stop and remove all services
down:
	@echo "Stopping all services..."
	docker-compose down
	@echo "All services stopped."

# Stop and remove all services including volumes
clean:
	@echo "Stopping all services and removing volumes..."
	docker-compose down -v
	@echo "All services stopped and volumes removed."

# Restart all services
restart: down up

# View logs
logs:
	docker-compose logs -f

# View logs for specific service
logs-%:
	docker-compose logs -f $*

# Show running services
ps:
	docker-compose ps

# Build images
build:
	@echo "Building Docker images..."
	docker-compose build
	@echo "Build complete!"

# Execute command in app container
exec:
	docker-compose exec app sh

# Check service health
health:
	@echo "Checking service health..."
	@docker-compose ps
	@echo "\nTesting Ethereum connection..."
	@curl -s -X POST -H "Content-Type: application/json" --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' http://localhost:8545 | jq .
	@echo "\nTesting Solana connection..."
	@curl -s http://localhost:8899 -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' | jq .