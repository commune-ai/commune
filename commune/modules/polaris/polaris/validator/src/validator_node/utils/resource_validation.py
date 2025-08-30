import logging
from typing import Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)

def calculate_cpu_score(specs: Dict[str, Any]) -> float:
    """Calculate the score for CPU resources."""
    try:
        cpu_count = int(specs.get('cpu_count', 0))
        cpu_speed = float(specs.get('cpu_speed', 0.0))
        
        # Basic CPU score calculation
        cpu_score = cpu_count * cpu_speed
        return min(10.0, cpu_score / 1000)  # Normalize to max 10
    except Exception as e:
        logger.error(f"Error calculating CPU score: {e}")
        return 0.0

def calculate_gpu_score(specs: Dict[str, Any]) -> float:
    """Calculate the score for GPU resources."""
    try:
        gpus = specs.get('gpus', [])
        if not gpus:
            return 0.0
        
        total_gpu_score = 0.0
        for gpu in gpus:
            # Extract GPU memory (in GB) and name
            memory = gpu.get('memory', 0)  # Memory in MB
            memory_gb = memory / 1024.0  # Convert to GB
            name = gpu.get('name', '').lower()
            
            # Base score on memory
            gpu_score = memory_gb * 0.5
            
            # Bonus for powerful GPUs
            if 'a100' in name:
                gpu_score *= 1.5
            elif 'h100' in name:
                gpu_score *= 2.0
            elif '3090' in name or '4090' in name:
                gpu_score *= 1.3
            
            total_gpu_score += gpu_score
        
        return min(20.0, total_gpu_score)  # Normalize to max 20
    except Exception as e:
        logger.error(f"Error calculating GPU score: {e}")
        return 0.0

def calculate_memory_score(specs: Dict[str, Any]) -> float:
    """Calculate the score for memory resources."""
    try:
        memory = float(specs.get('memory', 0))  # Memory in GB
        return min(5.0, memory / 10.0)  # Normalize to max 5
    except Exception as e:
        logger.error(f"Error calculating memory score: {e}")
        return 0.0

def calculate_storage_score(specs: Dict[str, Any]) -> float:
    """Calculate the score for storage resources."""
    try:
        storage = float(specs.get('storage', 0))  # Storage in GB
        return min(5.0, storage / 100.0)  # Normalize to max 5
    except Exception as e:
        logger.error(f"Error calculating storage score: {e}")
        return 0.0

def calculate_network_score(specs: Dict[str, Any]) -> float:
    """Calculate the score for network resources."""
    try:
        bandwidth = float(specs.get('bandwidth', 0))  # Bandwidth in Mbps
        return min(5.0, bandwidth / 100.0)  # Normalize to max 5
    except Exception as e:
        logger.error(f"Error calculating network score: {e}")
        return 0.0

def calculate_container_usage(container_data: Dict[str, Any]) -> float:
    """Calculate the score based on container usage."""
    try:
        # Calculate based on active time and utilization
        active_time = container_data.get('active_time', 0)  # In minutes
        cpu_utilization = container_data.get('cpu_utilization', 0)  # In percentage
        memory_utilization = container_data.get('memory_utilization', 0)  # In percentage
        
        # Basic formula: active_time * (cpu_util + memory_util) / 200
        # This gives higher scores to containers that are used more
        usage_score = active_time * (cpu_utilization + memory_utilization) / 200.0
        return min(10.0, usage_score)  # Normalize to max 10
    except Exception as e:
        logger.error(f"Error calculating container usage score: {e}")
        return 0.0

def validate_miner_resources(claimed_specs: Dict[str, Any], actual_specs: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that claimed resources match actual resources."""
    try:
        # Check CPU
        if int(claimed_specs.get('cpu_count', 0)) > int(actual_specs.get('cpu_count', 0)) * 1.1:
            return False, "Claimed CPU count significantly exceeds actual count"
        
        # Check GPU count
        claimed_gpus = claimed_specs.get('gpus', [])
        actual_gpus = actual_specs.get('gpus', [])
        if len(claimed_gpus) > len(actual_gpus):
            return False, "Claimed GPU count exceeds actual count"
        
        # Check memory (allow 10% tolerance)
        if float(claimed_specs.get('memory', 0)) > float(actual_specs.get('memory', 0)) * 1.1:
            return False, "Claimed memory significantly exceeds actual memory"
        
        # All checks passed
        return True, "Resource validation passed"
    except Exception as e:
        logger.error(f"Error validating miner resources: {e}")
        return False, f"Resource validation error: {str(e)}" 