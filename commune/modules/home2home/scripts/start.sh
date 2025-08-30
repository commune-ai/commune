 # start of file
#!/bin/bash

echo "Starting Home2Home application..."

# Build and start the Docker containers
docker-compose up -d

echo "Waiting for services to initialize..."
sleep 10

echo "Home2Home application is now running!"
echo "Frontend: http://localhost:3000"
echo "Local Ethereum Network: http://localhost:8545"

echo "To view logs, run: docker-compose logs -f"
echo "To stop the application, run: docker-compose down"
