
import docker
from datetime import datetime
import logging
from typing import Dict, List

class ContainerCostTracker:
    def __init__(self):
        self.client = docker.from_client()
        self.user_containers: Dict[str, List[str]] = {}
        self.container_costs: Dict[str, float] = {}
        self.cost_per_hour = 1.0  # Default cost per hour
        self.logger = logging.getLogger(__name__)

    def start_container(self, user_id: str, image: str) -> str:
        """Start a new container for a user."""
        try:
            container = self.client.containers.run(
                image,
                detach=True,
                labels={'user_id': user_id, 'start_time': str(datetime.now())}
            )
            
            if user_id not in self.user_containers:
                self.user_containers[user_id] = []
            
            self.user_containers[user_id].append(container.id)
            return container.id
        
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to start container: {e}")
            raise

    def stop_container(self, container_id: str) -> None:
        """Stop a container and calculate its cost."""
        try:
            container = self.client.containers.get(container_id)
            start_time = datetime.fromisoformat(container.labels['start_time'])
            duration = (datetime.now() - start_time).total_seconds() / 3600
            
            cost = duration * self.cost_per_hour
            self.container_costs[container_id] = cost
            
            container.stop()
            
        except docker.errors.DockerException as e:
            self.logger.error(f"Failed to stop container: {e}")
            raise

    def get_user_costs(self, user_id: str) -> float:
        """Calculate total costs for a user."""
        total_cost = 0.0
        if user_id in self.user_containers:
            for container_id in self.user_containers[user_id]:
                if container_id in self.container_costs:
                    total_cost += self.container_costs[container_id]
        return total_cost

    def list_user_containers(self, user_id: str) -> List[Dict]:
        """List all containers for a user with their details."""
        containers = []
        if user_id in self.user_containers:
            for container_id in self.user_containers[user_id]:
                try:
                    container = self.client.containers.get(container_id)
                    containers.append({
                        'id': container_id,
                        'status': container.status,
                        'cost': self.container_costs.get(container_id, 0.0)
                    })
                except docker.errors.DockerException:
                    continue
        return containers
