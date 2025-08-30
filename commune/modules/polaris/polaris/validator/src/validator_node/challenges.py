# src/neurons/Validator/challenges.py
import logging
import random
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ChallengeGenerator:
    def __init__(self):
        self.active_challenges = {}

    def generate_challenge(self, container_id: str) -> Dict[str, Any]:
        try:
            challenge_type = random.choice(['compute', 'memory'])
            
            if challenge_type == 'compute':
                return self._generate_compute_challenge()
            else:
                return self._generate_memory_challenge()

        except Exception as e:
            logger.error(f"Challenge generation failed: {str(e)}")
            return {"status": "error", "message": str(e)}
          
    def _generate_compute_challenge(self) -> Dict[str, Any]:
        return {
            "type": "compute",
            "data": {
                "command": "stress-ng --cpu 2 --cpu-method all --timeout 15s",
                "duration": 15,
                "expected_cpu": 80.0
            }
        }

    def _generate_memory_challenge(self) -> Dict[str, Any]:
        return {
            "type": "memory",
            "data": {
                "command": "stress-ng --vm 2 --vm-bytes 256M --vm-method all --timeout 15s",
                "duration": 15,
                "expected_memory": 256 * 1024 * 1024
            }
        }