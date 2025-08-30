import logging
import os
from typing import Dict, List, Optional
import firebase_admin
from firebase_admin import credentials, firestore

logger = logging.getLogger(__name__)

class FirebaseClient:
    """Client for interacting with Firebase to fetch miner data."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create a singleton instance of the Firebase client."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the Firebase client with credentials."""
        try:
            # Check if already initialized
            if not firebase_admin._apps:
                # Path to service account credentials
                cred_path = os.environ.get('FIREBASE_CREDS_PATH', 'firebase-creds.json')
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logger.info("Firebase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase client: {e}")
            raise
    
    def get_registered_miners(self) -> Dict[str, Dict]:
        """Fetch all registered miners from Firebase."""
        miners = {}
        try:
            miners_ref = self.db.collection('miners')
            docs = miners_ref.stream()
            
            for doc in docs:
                miner_data = doc.to_dict()
                # Add the document ID as miner_id
                miner_data['miner_id'] = doc.id
                miners[doc.id] = miner_data
                
            logger.info(f"Fetched {len(miners)} registered miners from Firebase")
        except Exception as e:
            logger.error(f"Error fetching registered miners: {e}")
        
        return miners
    
    def get_network_miners(self, network: str) -> Dict[str, Dict]:
        """Fetch miners registered for a specific network."""
        miners = {}
        try:
            miners_ref = self.db.collection('miners').where('network', '==', network)
            docs = miners_ref.stream()
            
            for doc in docs:
                miner_data = doc.to_dict()
                # Add the document ID as miner_id
                miner_data['miner_id'] = doc.id
                miners[doc.id] = miner_data
                
            logger.info(f"Fetched {len(miners)} {network} miners from Firebase")
        except Exception as e:
            logger.error(f"Error fetching {network} miners: {e}")
        
        return miners
    
    def update_miner_status(self, miner_id: str, status: str, verification_result: Optional[Dict] = None) -> bool:
        """Update the status of a miner in Firebase."""
        try:
            miner_ref = self.db.collection('miners').document(miner_id)
            update_data = {'status': status}
            
            if verification_result:
                update_data['verification_result'] = verification_result
                update_data['last_verified'] = firestore.SERVER_TIMESTAMP
            
            miner_ref.update(update_data)
            logger.info(f"Updated miner {miner_id} status to '{status}'")
            return True
        except Exception as e:
            logger.error(f"Error updating miner {miner_id} status: {e}")
            return False
    
    def save_validation_results(self, miner_id: str, results: Dict) -> bool:
        """Save validation results for a miner."""
        try:
            # Create a new validation history entry
            validation_ref = self.db.collection('miners').document(miner_id).collection('validations').document()
            results['timestamp'] = firestore.SERVER_TIMESTAMP
            validation_ref.set(results)
            
            # Update the miner's latest score
            if 'score' in results:
                miner_ref = self.db.collection('miners').document(miner_id)
                miner_ref.update({
                    'latest_score': results['score'],
                    'last_validated': firestore.SERVER_TIMESTAMP
                })
            
            logger.info(f"Saved validation results for miner {miner_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving validation results for miner {miner_id}: {e}")
            return False 