
# Docker Container Cost Tracker

A Python-based system to manage Docker containers and track container usage costs per user.

## Features
- Track Docker container usage per user
- Calculate costs based on runtime and resource usage
- Generate usage reports
- Manage container lifecycle

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/docker-cost-tracker.git

# Build the Docker environment
./scripts/build.sh

# Run the application
./scripts/run.sh
```

## Usage
```python
from docker_cost_tracker import ContainerCostTracker

tracker = ContainerCostTracker()
tracker.start_container("user1", "nginx")
# ... use container ...
tracker.stop_container("container_id")
costs = tracker.get_user_costs("user1")
```

## Testing
```bash
pytest tests/
```
