import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .models.schemas import ContainerRequest, ContainerResponse
from .services.container import ContainerManager
from .services.ssh import SSHManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Container Service API")
container_manager = ContainerManager()
ssh_manager = SSHManager()

@app.post("/containers", response_model=ContainerResponse)
async def create_container(request: ContainerRequest) -> Dict[str, Any]:
    """
    Create a new development container with SSH access.
    
    Args:
        request: Container creation request with SSH public key
    Returns:
        Container details including SSH access information
    """
    try:
        logger.info("Creating new container...")
        
        # Create container
        container_id = container_manager.create_container()
        logger.info(f"Container created with ID: {container_id}")
        
        # Setup SSH access
        ssh_details = ssh_manager.setup_ssh_access(
            container_id=container_id,
            public_key=request.public_key
        )
        logger.info(f"SSH access configured for container {container_id}")
        
        return ContainerResponse(
            container_id=container_id,
            ssh_port=ssh_details["port"],
            username=ssh_details["username"],
            host=ssh_details["host"]
        )
    except Exception as e:
        logger.error(f"Failed to create container: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/containers/{container_id}")
async def terminate_container(container_id: str) -> Dict[str, str]:
    """
    Terminate a container and remove its SSH access.
    
    Args:
        container_id: ID of the container to terminate
    Returns:
        Success message
    """
    try:
        logger.info(f"Terminating container {container_id}")
        
        # Remove SSH access first
        ssh_manager.remove_ssh_access(container_id)
        logger.info(f"SSH access removed for container {container_id}")
        
        # Then remove the container
        container_manager.remove_container(container_id)
        logger.info(f"Container {container_id} removed")
        
        return {
            "status": "success",
            "message": f"Container {container_id} terminated"
        }
    except Exception as e:
        logger.error(f"Failed to terminate container: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/containers/{container_id}", response_model=ContainerResponse)
async def get_container_info(container_id: str) -> Dict[str, Any]:
    """
    Get information about a specific container.
    
    Args:
        container_id: ID of the container
    Returns:
        Container details including SSH access information
    """
    try:
        logger.info(f"Retrieving information for container {container_id}")
        
        container_info = container_manager.get_container_info(container_id)
        ssh_info = ssh_manager.get_ssh_info(container_id)
        
        return ContainerResponse(
            container_id=container_id,
            ssh_port=ssh_info["port"],
            username=ssh_info["username"],
            host=ssh_info["host"]
        )
    except Exception as e:
        logger.error(f"Failed to get container info: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))