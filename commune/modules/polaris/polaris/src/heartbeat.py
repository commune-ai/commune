import json
import logging
import platform
import time
from typing import Any, Dict

import psutil
import requests

logger = logging.getLogger(__name__)

# Server configuration - get from environment or config later
SERVER_URL = "https://polaris-test-server.onrender.com/api/v1"

def get_system_metrics() -> Dict[str, Any]:
    """Collect basic system metrics for the heartbeat."""
    try:
        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "system": platform.system(),
            "uptime": time.time() - psutil.boot_time()
        }
        return metrics
    except Exception as e:
        logger.error(f"Error collecting system metrics: {str(e)}")
        return {"error": str(e)}

def send_heartbeat(miner_id: str) -> bool:
    """Send a heartbeat to the server."""
    try:
        metrics = get_system_metrics()
        
        data = {
            "miner_id": miner_id,
            "timestamp": time.time(),
            "status": "online",
            "metrics": metrics
        }
        
        # Currently just logging instead of sending
        logger.debug(f"Heartbeat data: {json.dumps(data)}")
        
        # Uncomment to actually send the heartbeat
        # response = requests.post(f"{SERVER_URL}/miners/{miner_id}/heartbeat", json=data)
        # return response.status_code == 200
        
        return True
    except Exception as e:
        logger.error(f"Error sending heartbeat: {str(e)}")
        return False

def start_heartbeat(miner_id: str, interval: int = 60):
    """Start sending heartbeats at regular intervals."""
    logger.info(f"Starting heartbeat service for miner {miner_id}")
    
    while True:
        try:
            success = send_heartbeat(miner_id)
            if success:
                logger.debug(f"Heartbeat sent successfully for miner {miner_id}")
            else:
                logger.warning(f"Failed to send heartbeat for miner {miner_id}")
        except Exception as e:
            logger.error(f"Error in heartbeat service: {str(e)}")
        
        # Sleep until next heartbeat
        time.sleep(interval) 