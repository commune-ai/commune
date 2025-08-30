"""
Core Validator Base Class

This module provides the base Validator class that defines the common interface
and shared functionality for all network-specific validator implementations.
"""

import asyncio
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional

from loguru import logger
from pydantic import BaseModel
from substrateinterface import Keypair

from validator.src.validator_node._config import ValidatorSettings
from validator.src.validator_node.base import BaseValidator


class WeightHistory(BaseModel):
    """Record of weight assignments for historical tracking"""
    time: datetime
    data: List  # List of (uid, weight) tuples
    network: str  # Which network these weights were for


class CoreValidator(ABC):
    """Base Validator class that defines the interface for all validator implementations.
    
    This abstract class defines the methods that must be implemented by network-specific
    validator classes, as well as providing common functionality that all validators share.
    """
    
    def __init__(self, key: Keypair, settings: ValidatorSettings | None = None):
        """Initialize the core validator.
        
        Args:
            key: The keypair used for authentication and signing
            settings: Validator settings (if None, default settings will be used)
        """
        self.key = key
        self.settings = settings or ValidatorSettings()
        self.weights_histories = deque(maxlen=10)
        
        # Initialize logging
        logger.info(f"Initializing {self.__class__.__name__}")

    @abstractmethod
    async def initialize_network(self) -> bool:
        """Initialize network-specific connections and state.
        
        This method should set up any network clients, retrieve network IDs,
        and perform other network-specific initialization.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def validate_step(self) -> None:
        """Perform a single validation step for all miners.
        
        This method should:
        1. Get all miners to validate
        2. Score each miner based on network-specific criteria
        3. Calculate appropriate weights
        4. Submit weights to the network
        """
        pass
    
    @abstractmethod
    async def get_miners(self) -> Dict[str, Any]:
        """Get the list of miners to validate.
        
        Returns:
            Dict: Dictionary of miners keyed by their ID
        """
        pass
    
    @abstractmethod
    async def score_miners(self, miners: Dict[str, Any]) -> Dict[str, float]:
        """Score each miner based on their performance and resources.
        
        Args:
            miners: Dictionary of miners to score
            
        Returns:
            Dict: Dictionary mapping miner IDs to their scores
        """
        pass
    
    @abstractmethod
    async def set_weights(self, weights: Dict[str, float]) -> bool:
        """Set weights for miners on the network.
        
        Args:
            weights: Dictionary mapping miner IDs to their assigned weights
            
        Returns:
            bool: True if weights were successfully set, False otherwise
        """
        pass
    
    def normalize_scores(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to a range of 0-1.
        
        Args:
            score_dict: Dictionary of raw scores
            
        Returns:
            Dict: Dictionary of normalized scores
        """
        if not score_dict:
            return {}
            
        # Find min and max scores
        min_score = min(score_dict.values())
        max_score = max(score_dict.values())
        
        # Handle edge case where all scores are the same
        if max_score == min_score:
            return {k: 1.0 for k in score_dict.keys()}
            
        # Normalize scores to 0-1 range
        normalized = {
            k: (v - min_score) / (max_score - min_score)
            for k, v in score_dict.items()
        }
        
        return normalized
    
    def assign_weight(self, score: float) -> int:
        """Scale normalized scores to the network's weight range.
        
        Args:
            score: Normalized score (0-1)
            
        Returns:
            int: Weight value for the network
        """
        max_score = 1.0  # Maximum normalized score
        weight = int(score * 5000 / max_score)  # Scale to 0-5000 range
        return weight
    
    def cut_to_max_allowed_weights(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """Trim weights to the maximum allowed count.
        
        Args:
            score_dict: Dictionary of scores to trim
            
        Returns:
            Dict: Trimmed dictionary with at most max_allowed_weights entries
        """
        max_allowed_weights = self.settings.max_allowed_weights
        
        if len(score_dict) > max_allowed_weights:
            sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            trimmed = dict(sorted_scores[:max_allowed_weights])
            logger.info(f"Trimmed scores from {len(score_dict)} to max allowed: {len(trimmed)}")
            return trimmed
        return score_dict
    
    def validation_loop(self):
        """Continuously validate miners."""
        logger.info(f"Starting validation loop for {self.__class__.__name__}")
        while True:
            try:
                asyncio.run(self.validate_step())
                logger.debug(f"Completed validation step, sleeping for {self.settings.iteration_interval}s")
                time.sleep(self.settings.iteration_interval)
            except Exception as e:
                logger.error(f"Error in validation loop: {e}")
                logger.error(traceback.format_exc())
                # Brief pause to avoid rapid-fire errors in case of persistent issues
                time.sleep(5)

    def start_validation_loop(self):
        """Start the validation loop in a separate thread."""
        logger.info(f"Starting validation loop thread for {self.__class__.__name__}")
        thread = threading.Thread(target=self.validation_loop, daemon=True)
        thread.start()
        return thread
    
    def add_weights_history(self, uids: List[str], weights: List[float], network: str):
        """Add a weight history entry.
        
        Args:
            uids: List of miner UIDs
            weights: List of corresponding weights
            network: Network name (e.g., "commune" or "bittensor")
        """
        weight_data = list(zip(uids, weights))
        self.weights_histories.append(
            WeightHistory(
                time=datetime.now(),
                data=weight_data,
                network=network
            )
        )
        logger.debug(f"Added weight history entry for {network} with {len(uids)} miners")
    
    def get_weights_history(self):
        """Retrieve the history of weights.
        
        Returns:
            List: List of WeightHistory objects
        """
        return list(self.weights_histories) 