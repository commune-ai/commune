
import requests

import json
import requests
import commune as c

class Inch(c.Module):

    description = """
        Gets token balances for a wallet address from the 1Inch Balance API.
        :param wallet_address: A wallet address.
        :return: A JSON blob with token balances.
    """
    def __init__(self, 
                 api_key: str = 'INCH_API_KEY',
                 url= "https://api.1inch.dev/balance"):

        self.api_key = api_key
        self.url = url

    def call(self, wallet_address:str ):
        endpoint = f'https://api.1inch.dev/balance/v1.2/1/balances/{wallet_address}'
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            # Send a GET request to the API
            response = requests.get(endpoint, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch data from 1Inch Balance API: {e}")
            return
    
        # Parse the JSON response
        response_data = response.json()
        json_blob = json.dumps(response_data, indent=4)  # Convert the Python dictionary to a JSON formatted string
        
        return json_blob
    

    @classmethod
    def test(cls):
        wallet_address = '0xbe0eb53f46cd790cd13851d5eff43d12404d33e8'
        tool = cls()

        token_balances = tool.call(wallet_address)

        if token_balances:
            print(f"Token balances for wallet address {wallet_address}:")
            token_balances_dict = json.loads(token_balances)
            print(token_balances_dict)
        else:
            print("Token balance fetch failed. Please check your wallet address.")
