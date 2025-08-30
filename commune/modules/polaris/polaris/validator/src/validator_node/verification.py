# src/neurons/Validator/verification.py
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self):
        self.verifications = {}

    def verify_resource_usage(self, container_id: str, result: Dict[str, Any]) -> bool:
        try:
            if result["status"] != "success":
                return False

            metrics = result.get("metrics", {})
            challenge_type = result.get("type")

            if challenge_type == "compute":
                return self._verify_compute_usage(metrics)
            elif challenge_type == "memory":
                return self._verify_memory_usage(metrics)
            return False

        except Exception as e:
            logger.error(f"Resource verification failed: {str(e)}")
            return False

    def _verify_compute_usage(self, metrics: Dict[str, Any]) -> bool:
        cpu_usage = metrics.get("cpu_usage", 0)
        # Verify CPU usage is reasonable (above 50%)
        return cpu_usage > 50.0

    def _verify_memory_usage(self, metrics: Dict[str, Any]) -> bool:
        try:
            memory_usage = metrics.get("memory_usage", 0)
            memory_limit = metrics.get("memory_limit", 0)
            memory_percent = metrics.get("memory_percent", 0)
            
            # Consider a minimum of 15% memory usage as valid
            # This accounts for some overhead and variation
            return memory_percent >= 15.0
        except Exception as e:
            logger.error(f"Memory verification failed: {str(e)}")
            return False