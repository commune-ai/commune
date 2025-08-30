import requests
import json
import os

class IPFSClient:
    def __init__(self, api_url='http://127.0.0.1:5001/api/v0'):
        self.api_url = api_url.rstrip('/')

    def _post(self, endpoint, files=None, params=None):
        url = f"{self.api_url}/{endpoint}"
        try:
            res = requests.post(url, files=files, params=params)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            print(f"❌ IPFS Error: {e}")
            return None

    def add_file(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        with open(filepath, 'rb') as f:
            files = {'file': f}
            return self._post('add', files=files)

    def add_data(self, data: str):
        files = {'file': ('data.txt', data)}
        return self._post('add', files=files)

    def cat(self, cid):
        url = f"{self.api_url}/cat?arg={cid}"
        try:
            res = requests.post(url)
            res.raise_for_status()
            return res.text
        except Exception as e:
            print(f"❌ Cat Error: {e}")
            return None

    def pin_add(self, cid):
        return self._post('pin/add', params={'arg': cid})

    def pin_rm(self, cid):
        return self._post('pin/rm', params={'arg': cid})

    def ls_pins(self):
        return self._post('pin/ls')

# 🧪 Example usage
if __name__ == "__main__":
    ipfs = IPFSClient()

    # Add data
    result = ipfs.add_data("gm fren 🧠")
    print("📦 Added data CID:", result.get('Hash'))

    # Retrieve
    print("📖 Fetched content:", ipfs.cat(result.get('Hash')))

    # Pin it
    ipfs.pin_add(result.get('Hash'))
