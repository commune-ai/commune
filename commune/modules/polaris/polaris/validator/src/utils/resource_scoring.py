"""
Resource scoring utilities for the Polaris validator system.

This module contains functions for calculating scores for various hardware
components (CPU, GPU, memory, storage, network) and container usage. These
scores are used to determine miners' overall weight in the network.
"""
import logging
from typing import Dict, Any, List, Tuple, Optional

from validator.src.config import ScoringConfig

logger = logging.getLogger(__name__)

def calculate_cpu_score(specs: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate the score for CPU resources.
    
    Args:
        specs: Hardware specifications containing CPU details
        config: Scoring configuration
        
    Returns:
        A score for the CPU resources (0.0 to config.cpu_max_score)
    """
    try:
        # Extract CPU details
        cpu_count = int(specs.get('cpu_count', 0))
        cpu_speed = float(specs.get('cpu_speed', 0.0))  # In GHz
        
        # Basic score calculation: CPU cores * speed (GHz) / normalization factor
        raw_score = cpu_count * cpu_speed / config.cpu_normalization_factor
        
        # Apply max limit
        return min(config.cpu_max_score, raw_score * config.cpu_max_score)
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Error calculating CPU score: {e}")
        return 0.0

def calculate_gpu_score(specs: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate the score for GPU resources.
    
    Args:
        specs: Hardware specifications containing GPU details
        config: Scoring configuration
        
    Returns:
        A score for the GPU resources (0.0 to config.gpu_max_score)
    """
    try:
        gpus = specs.get('gpus', [])
        if not gpus:
            return 0.0
        
        total_gpu_score = 0.0
        
        for gpu in gpus:
            # Extract GPU details
            memory_mb = gpu.get('memory', 0)  # Memory in MB
            memory_gb = memory_mb / 1024.0  # Convert to GB
            name = str(gpu.get('name', '')).lower()
            
            # Base score based on memory
            gpu_score = memory_gb * config.gpu_base_factor
            
            # Apply bonus for specific GPU types
            for gpu_type, bonus_factor in config.gpu_bonus_factors.items():
                if gpu_type in name:
                    gpu_score *= bonus_factor
                    break
            
            total_gpu_score += gpu_score
        
        # Apply max limit
        return min(config.gpu_max_score, total_gpu_score)
    
    except Exception as e:
        logger.warning(f"Error calculating GPU score: {e}")
        return 0.0

def calculate_memory_score(specs: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate the score for memory resources.
    
    Args:
        specs: Hardware specifications containing memory details
        config: Scoring configuration
        
    Returns:
        A score for the memory resources (0.0 to config.memory_max_score)
    """
    try:
        # Extract memory details
        memory_gb = float(specs.get('memory', 0))  # Memory in GB
        
        # Score calculation: memory (GB) / normalization factor
        raw_score = memory_gb / config.memory_normalization_factor
        
        # Apply max limit
        return min(config.memory_max_score, raw_score)
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Error calculating memory score: {e}")
        return 0.0

def calculate_storage_score(specs: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate the score for storage resources.
    
    Args:
        specs: Hardware specifications containing storage details
        config: Scoring configuration
        
    Returns:
        A score for the storage resources (0.0 to config.storage_max_score)
    """
    try:
        # Extract storage details
        storage_gb = float(specs.get('storage', 0))  # Storage in GB
        
        # Score calculation: storage (GB) / normalization factor
        raw_score = storage_gb / config.storage_normalization_factor
        
        # Apply max limit
        return min(config.storage_max_score, raw_score)
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Error calculating storage score: {e}")
        return 0.0

def calculate_network_score(specs: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate the score for network resources.
    
    Args:
        specs: Hardware specifications containing network details
        config: Scoring configuration
        
    Returns:
        A score for the network resources (0.0 to config.network_max_score)
    """
    try:
        # Extract network details
        bandwidth_mbps = float(specs.get('bandwidth', 0))  # Bandwidth in Mbps
        
        # Score calculation: bandwidth (Mbps) / normalization factor
        raw_score = bandwidth_mbps / config.network_normalization_factor
        
        # Apply max limit
        return min(config.network_max_score, raw_score)
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Error calculating network score: {e}")
        return 0.0

def calculate_hardware_score(specs: Dict[str, Any], config: ScoringConfig) -> Dict[str, float]:
    """
    Calculate scores for all hardware components.
    
    Args:
        specs: Hardware specifications containing all resource details
        config: Scoring configuration
        
    Returns:
        A dictionary with individual and total scores for hardware resources
    """
    # Calculate individual scores
    cpu_score = calculate_cpu_score(specs, config)
    gpu_score = calculate_gpu_score(specs, config)
    memory_score = calculate_memory_score(specs, config)
    storage_score = calculate_storage_score(specs, config)
    network_score = calculate_network_score(specs, config)
    
    # Calculate total hardware score
    total_score = cpu_score + gpu_score + memory_score + storage_score + network_score
    
    return {
        'cpu': cpu_score,
        'gpu': gpu_score,
        'memory': memory_score,
        'storage': storage_score,
        'network': network_score,
        'total': total_score
    }

def calculate_container_usage_score(container_data: Dict[str, Any], config: ScoringConfig) -> float:
    """
    Calculate a score based on container usage metrics.
    
    Args:
        container_data: Container usage metrics
        config: Scoring configuration
        
    Returns:
        A score for container usage (0.0 to config.container_max_score)
    """
    try:
        # Extract container usage metrics
        active_time_minutes = float(container_data.get('active_time', 0))
        cpu_utilization_pct = float(container_data.get('cpu_utilization', 0))
        memory_utilization_pct = float(container_data.get('memory_utilization', 0))
        
        # Normalize utilization percentages (0-100 to 0-1)
        cpu_util = cpu_utilization_pct / 100.0
        mem_util = memory_utilization_pct / 100.0
        
        # Calculate score based on active time and utilization
        # Formula: active_time * (cpu_util + mem_util) / 2
        # This gives higher scores to containers that are used more intensively for longer periods
        utilization_factor = (cpu_util + mem_util) / 2.0
        
        # Limit active time to a reasonable maximum (e.g., 24 hours = 1440 minutes)
        capped_active_time = min(active_time_minutes, 1440)
        
        # Calculate score (normalize to max score)
        raw_score = capped_active_time * utilization_factor / 1440.0
        
        # Apply max limit
        return min(config.container_max_score, raw_score * config.container_max_score)
    
    except (ValueError, TypeError, ZeroDivisionError) as e:
        logger.warning(f"Error calculating container usage score: {e}")
        return 0.0

def calculate_container_scores(containers: List[Dict[str, Any]], config: ScoringConfig) -> Dict[str, float]:
    """
    Calculate scores for container usage.
    
    Args:
        containers: List of container usage data
        config: Scoring configuration
        
    Returns:
        A dictionary with container scores and average score
    """
    if not containers:
        return {'containers': {}, 'average': 0.0, 'count': 0}
    
    container_scores = {}
    total_score = 0.0
    
    for container in containers:
        container_id = container.get('id', 'unknown')
        score = calculate_container_usage_score(container, config)
        container_scores[container_id] = score
        total_score += score
    
    average_score = total_score / len(containers)
    
    return {
        'containers': container_scores,
        'average': average_score,
        'count': len(containers)
    }

def calculate_miner_score(
    hardware_specs: Dict[str, Any],
    container_data: List[Dict[str, Any]],
    config: ScoringConfig
) -> Dict[str, Any]:
    """
    Calculate the overall score for a miner based on hardware and container usage.
    
    Args:
        hardware_specs: Hardware specifications of the miner
        container_data: Container usage data for the miner
        config: Scoring configuration
        
    Returns:
        A dictionary with detailed scoring information
    """
    # Calculate hardware scores
    hw_scores = calculate_hardware_score(hardware_specs, config)
    
    # Calculate container scores
    container_scores = calculate_container_scores(container_data, config)
    
    # Calculate final score using weighted average
    hw_weight = config.hardware_weight
    container_weight = config.container_weight
    
    # If there are no containers, use only hardware score
    if container_scores['count'] == 0:
        final_score = hw_scores['total']
    else:
        final_score = (hw_scores['total'] * hw_weight) + (container_scores['average'] * container_weight)
    
    return {
        'hardware': hw_scores,
        'containers': container_scores,
        'final_score': final_score,
        'components': {
            'hardware_contribution': hw_scores['total'] * hw_weight,
            'container_contribution': container_scores['average'] * container_weight
        }
    }

def validate_miner_resources(
    claimed_specs: Dict[str, Any],
    actual_specs: Dict[str, Any],
    tolerance: float = 0.1
) -> Tuple[bool, str]:
    """
    Validate that the resources claimed by a miner match the actual resources detected.
    
    Args:
        claimed_specs: Resources claimed by the miner during registration
        actual_specs: Resources detected during verification
        tolerance: Fractional tolerance for resource matching (0.1 = 10%)
        
    Returns:
        A tuple of (is_valid, reason_if_invalid)
    """
    # Validate CPU count
    claimed_cpu = int(claimed_specs.get('cpu_count', 0))
    actual_cpu = int(actual_specs.get('cpu_count', 0))
    
    if claimed_cpu > actual_cpu * (1 + tolerance):
        return False, f"Claimed CPU count ({claimed_cpu}) exceeds actual count ({actual_cpu}) by more than {tolerance*100}%"
    
    # Validate GPU count
    claimed_gpus = claimed_specs.get('gpus', [])
    actual_gpus = actual_specs.get('gpus', [])
    
    if len(claimed_gpus) > len(actual_gpus):
        return False, f"Claimed GPU count ({len(claimed_gpus)}) exceeds actual count ({len(actual_gpus)})"
    
    # Validate memory
    claimed_memory = float(claimed_specs.get('memory', 0))
    actual_memory = float(actual_specs.get('memory', 0))
    
    if claimed_memory > actual_memory * (1 + tolerance):
        return False, f"Claimed memory ({claimed_memory:.1f} GB) exceeds actual memory ({actual_memory:.1f} GB) by more than {tolerance*100}%"
    
    # Validate storage
    claimed_storage = float(claimed_specs.get('storage', 0))
    actual_storage = float(actual_specs.get('storage', 0))
    
    if claimed_storage > actual_storage * (1 + tolerance):
        return False, f"Claimed storage ({claimed_storage:.1f} GB) exceeds actual storage ({actual_storage:.1f} GB) by more than {tolerance*100}%"
    
    # All checks passed
    return True, "Resource validation passed" 