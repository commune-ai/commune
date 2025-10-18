import os
import json
from typing import Optional, Dict, Any, List
import requests

class ArweaveStore:
    """Arweave decentralized storage implementation"""
    
    def __init__(self, gateway: str = "https://arweave.net", wallet_path: Optional[str] = None):
        self.gateway = gateway
        self.wallet_path = wallet_path
        self.wallet = self._load_wallet() if wallet_path else None
    
    def _load_wallet(self) -> Optional[Dict]:
        """Load Arweave wallet from file"""
        if self.wallet_path and os.path.exists(self.wallet_path):
            with open(self.wallet_path, 'r') as f:
                return json.load(f)
        return None
    
    def write(self, data: bytes, tags: Optional[List[Dict[str, str]]] = None) -> str:
        """Write data to Arweave and return transaction ID"""
        # Implementation would use arweave-python-client
        raise NotImplementedError("Requires arweave-python-client library")
    
    def read(self, tx_id: str) -> bytes:
        """Read data from Arweave by transaction ID"""
        response = requests.get(f"{self.gateway}/{tx_id}")
        response.raise_for_status()
        return response.content
    
    def exists(self, tx_id: str) -> bool:
        """Check if transaction exists"""
        try:
            response = requests.head(f"{self.gateway}/{tx_id}")
            return response.status_code == 200
        except:
            return False
    
    def get_metadata(self, tx_id: str) -> Dict[str, Any]:
        """Get transaction metadata"""
        response = requests.get(f"{self.gateway}/tx/{tx_id}")
        response.raise_for_status()
        return response.json()
    
    def list_transactions(self, address: str, limit: int = 100) -> List[str]:
        """List transactions for an address"""
        response = requests.get(f"{self.gateway}/graphql", json={
            "query": f"""query {{
                transactions(owners: [\"{address}\"], first: {limit}) {{
                    edges {{
                        node {{
                            id
                        }}
                    }}
                }}
            }}"""
        })
        data = response.json()
        return [edge['node']['id'] for edge in data['data']['transactions']['edges']]
