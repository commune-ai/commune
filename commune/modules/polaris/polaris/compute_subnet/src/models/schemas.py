from pydantic import BaseModel, Field, validator
import re
from typing import Optional

class ContainerRequest(BaseModel):
    """Request model for container creation"""
    public_key: str = Field(
        ...,
        description="SSH public key for container access"
    )

    @validator('public_key')
    def validate_ssh_key(cls, v):
        """Validate SSH public key format"""
        if not v.strip():
            raise ValueError("SSH public key cannot be empty")
        
        key_parts = v.split()
        if len(key_parts) < 2:
            raise ValueError("Invalid SSH key format")
            
        valid_types = [
            'ssh-rsa',
            'ssh-ed25519',
            'ecdsa-sha2-nistp256',
            'ecdsa-sha2-nistp384',
            'ecdsa-sha2-nistp521'
        ]
        
        if key_parts[0] not in valid_types:
            raise ValueError(f"Invalid SSH key type. Must be one of: {', '.join(valid_types)}")
            
        # Basic validation of the key data
        if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', key_parts[1]):
            raise ValueError("Invalid SSH key data format")
            
        return v

class ContainerResponse(BaseModel):
    """Response model for container operations"""
    container_id: str = Field(
        ...,
        description="Unique identifier for the container"
    )
    ssh_port: int = Field(
        ...,
        description="SSH port number for accessing the container"
    )
    username: str = Field(
        ...,
        description="Username for SSH access"
    )
    host: str = Field(
        ...,
        description="Hostname or IP address for SSH access"
    )
    status: Optional[str] = Field(
        None,
        description="Current status of the container"
    )

    @validator('ssh_port')
    def validate_port(cls, v):
        """Validate port number"""
        if not 1024 <= v <= 65535:
            raise ValueError("Port number must be between 1024 and 65535")
        return v