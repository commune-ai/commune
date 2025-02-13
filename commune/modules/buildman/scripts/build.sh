
#!/bin/bash
set -e

# Build Docker image
docker build -t docker-cost-tracker -f Dockerfile .

# Install Python dependencies
pip install -r requirements.txt
