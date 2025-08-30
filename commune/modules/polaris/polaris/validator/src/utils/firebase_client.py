"""
Firebase client utilities for Polaris validator system.

This module provides a standardized client for interacting with Firebase,
including fetching miner data and container information.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional, Union

import firebase_admin
from firebase_admin import credentials, firestore

from validator.src.utils.logging_utils import exception_handler

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Client for interacting with Firebase Firestore database."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the Firebase client.
        
        Args:
            credentials_path: Path to the Firebase credentials file.
                If None, will try to use environment variable FIREBASE_CREDENTIALS_PATH,
                or fallback to 'firebase_credentials.json' in the current directory.
        """
        self.app = None
        self.db = None
        self.initialized = False
        
        # Get credentials path
        if not credentials_path:
            credentials_path = os.environ.get(
                'FIREBASE_CREDENTIALS_PATH', 
                'firebase_credentials.json'
            )
        
        self.credentials_path = credentials_path
    
    def initialize(self) -> bool:
        """
        Initialize the Firebase app and Firestore client.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        if self.initialized:
            return True
        
        try:
            # Check if the credentials file exists
            if not os.path.exists(self.credentials_path):
                logger.error(f"Firebase credentials file not found at: {self.credentials_path}")
                return False
            
            # Initialize Firebase app
            cred = credentials.Certificate(self.credentials_path)
            self.app = firebase_admin.initialize_app(cred)
            
            # Initialize Firestore client
            self.db = firestore.client()
            self.initialized = True
            
            logger.info("Firebase client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase client: {e}")
            return False
    
    @exception_handler(fallback_return={})
    def get_registered_miners(self, network: Optional[str] = None) -> Dict[str, Any]:
        """
        Get miners registered on the Polaris platform.
        
        Args:
            network: Optional network filter (e.g., 'bittensor', 'commune')
                If provided, only miners registered for this network will be returned.
        
        Returns:
            A dictionary mapping miner IDs to their registration data
        """
        if not self.initialize():
            logger.error("Cannot get registered miners: Firebase not initialized")
            return {}
        
        miners_collection = self.db.collection('miners')
        
        # Apply network filter if provided
        if network:
            miners_query = miners_collection.where('network', '==', network)
        else:
            miners_query = miners_collection
        
        miners_data = {}
        
        try:
            # Execute query and process results
            for doc in miners_query.stream():
                miner_id = doc.id
                miner_data = doc.to_dict()
                miners_data[miner_id] = miner_data
            
            logger.info(f"Retrieved {len(miners_data)} registered miners" + 
                       (f" for network '{network}'" if network else ""))
            
            return miners_data
            
        except Exception as e:
            logger.error(f"Error retrieving registered miners: {e}")
            return {}
    
    @exception_handler(fallback_return={})
    def get_miner_containers(self, miner_id: str) -> List[Dict[str, Any]]:
        """
        Get container data for a specific miner.
        
        Args:
            miner_id: The ID of the miner
        
        Returns:
            A list of container data dictionaries for the specified miner
        """
        if not self.initialize():
            logger.error("Cannot get miner containers: Firebase not initialized")
            return []
        
        containers_collection = self.db.collection('containers')
        query = containers_collection.where('miner_id', '==', miner_id)
        
        container_data = []
        
        try:
            # Execute query and process results
            for doc in query.stream():
                container = doc.to_dict()
                container['id'] = doc.id
                container_data.append(container)
            
            logger.info(f"Retrieved {len(container_data)} containers for miner {miner_id}")
            
            return container_data
            
        except Exception as e:
            logger.error(f"Error retrieving containers for miner {miner_id}: {e}")
            return []
    
    @exception_handler(fallback_return=False)
    def update_miner_status(self, miner_id: str, status: str, 
                           verification_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of a miner in the Firestore database.
        
        Args:
            miner_id: The ID of the miner to update
            status: The new status value ('verified', 'unverified', 'pending', etc.)
            verification_data: Optional additional data about the verification
        
        Returns:
            True if the update was successful, False otherwise
        """
        if not self.initialize():
            logger.error("Cannot update miner status: Firebase not initialized")
            return False
        
        try:
            miner_ref = self.db.collection('miners').document(miner_id)
            
            # Prepare update data
            update_data = {
                'status': status,
                'last_verified': firestore.SERVER_TIMESTAMP
            }
            
            # Add verification data if provided
            if verification_data:
                update_data['verification_data'] = verification_data
            
            # Update the document
            miner_ref.update(update_data)
            
            logger.info(f"Updated miner {miner_id} status to '{status}'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating miner {miner_id} status: {e}")
            return False
    
    @exception_handler(fallback_return=False)
    def log_verification_result(self, miner_id: str, 
                               verification_result: Dict[str, Any]) -> bool:
        """
        Log the verification result for a miner.
        
        Args:
            miner_id: The ID of the miner
            verification_result: Data about the verification result
        
        Returns:
            True if logging was successful, False otherwise
        """
        if not self.initialize():
            logger.error("Cannot log verification result: Firebase not initialized")
            return False
        
        try:
            # Add a timestamp to the verification result
            verification_result['timestamp'] = firestore.SERVER_TIMESTAMP
            verification_result['miner_id'] = miner_id
            
            # Add to verification_logs collection
            self.db.collection('verification_logs').add(verification_result)
            
            logger.info(f"Logged verification result for miner {miner_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging verification result for miner {miner_id}: {e}")
            return False
    
    @exception_handler(fallback_return=False)
    def log_weight_submission(self, network: str, weights_data: Dict[str, Any]) -> bool:
        """
        Log a weight submission event.
        
        Args:
            network: The network for which weights were submitted
            weights_data: Data about the weight submission
        
        Returns:
            True if logging was successful, False otherwise
        """
        if not self.initialize():
            logger.error("Cannot log weight submission: Firebase not initialized")
            return False
        
        try:
            # Add network and timestamp information
            weights_data['network'] = network
            weights_data['timestamp'] = firestore.SERVER_TIMESTAMP
            
            # Add to weight_submissions collection
            self.db.collection('weight_submissions').add(weights_data)
            
            logger.info(f"Logged weight submission for {network} network")
            return True
            
        except Exception as e:
            logger.error(f"Error logging weight submission for {network} network: {e}")
            return False 