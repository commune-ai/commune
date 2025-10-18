import os
import json
from typing import Optional, Dict, Any, List, Union
import requests

class IPFSStore:
    """IPFS decentralized storage implementation"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5001, gateway: str = "https://ipfs.io"):
        self.api_url = f"http://{host}:{port}/api/v0"
        self.gateway = gateway
    
    def write(self, data: Union[bytes, str], path: Optional[str] = None) -> str:
        """Add data to IPFS and return CID"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        files = {'file': data}
        params = {}
        if path:
            params['wrap-with-directory'] = 'true'
            params['path'] = path
        
        response = requests.post(f"{self.api_url}/add", files=files, params=params)
        response.raise_for_status()
        result = response.json()
        return result['Hash']
    
    def read(self, cid: str) -> bytes:
        """Read data from IPFS by CID"""
        response = requests.post(f"{self.api_url}/cat", params={'arg': cid})
        response.raise_for_status()
        return response.content
    
    def read_via_gateway(self, cid: str) -> bytes:
        """Read data via public gateway"""
        response = requests.get(f"{self.gateway}/ipfs/{cid}")
        response.raise_for_status()
        return response.content
    
    def exists(self, cid: str) -> bool:
        """Check if CID exists locally"""
        try:
            response = requests.post(f"{self.api_url}/block/stat", params={'arg': cid}, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def pin(self, cid: str) -> bool:
        """Pin content to prevent garbage collection"""
        response = requests.post(f"{self.api_url}/pin/add", params={'arg': cid})
        return response.status_code == 200
    
    def unpin(self, cid: str) -> bool:
        """Unpin content"""
        response = requests.post(f"{self.api_url}/pin/rm", params={'arg': cid})
        return response.status_code == 200
    
    def list_pins(self) -> List[str]:
        """List all pinned CIDs"""
        response = requests.post(f"{self.api_url}/pin/ls")
        response.raise_for_status()
        result = response.json()
        return list(result.get('Keys', {}).keys())
    
    def get_stats(self, cid: str) -> Dict[str, Any]:
        """Get object statistics"""
        response = requests.post(f"{self.api_url}/object/stat", params={'arg': cid})
        response.raise_for_status()
        return response.json()
    
    def mkdir(self, path: str) -> str:
        """Create directory in IPFS MFS"""
        response = requests.post(f"{self.api_url}/files/mkdir", params={'arg': path, 'parents': 'true'})
        response.raise_for_status()
        return path
    
    def ls(self, path: str = '/') -> List[Dict[str, Any]]:
        """List directory contents in MFS"""
        response = requests.post(f"{self.api_url}/files/ls", params={'arg': path, 'long': 'true'})
        response.raise_for_status()
        result = response.json()
        return result.get('Entries', [])

    def run_local_node(self):
        """Utility to run a local IPFS node (requires ipfs installed)"""
        if os.system("which ipfs") != 0:
            # Install IPFS if not present
            self.install_ipfs()
        os.system("ipfs init")
        os.system("ipfs daemon &")

    def ipfs_installed(self) -> bool:
        """Check if IPFS command line tool is installed"""
        return os.system("which ipfs") == 0
    