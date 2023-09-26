import commune as c
import requests
import json
import os

class SwaggerInch(c.Module):
    def __init__(self, api_key: str = 'INCH_API_KEY'):
        self.api_key = os.getenv(api_key, api_key)
        
    description = """
    Gets the token prices from the 1inch API.
    """
    def call(self):
        """
        Connects to the 1Inch API for gas price
        """
        url = 'https://api.1inch.dev/gas-price/v1.4/1'
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            # Send a GET request to the API
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch data from 1Inch Gas Price API: {e}")
            return
    
        # Parse the JSON response
        response_data = response.json()
        # json_blob = json.dumps(response_data, indent=4)  # Convert the Python dictionary to a JSON formatted string
        # print(json_blob)
        return response_data["baseFee"]
    

if __name__ == "__main__":
     dl_instance = SwaggerInch()
     result=dl_instance.call()
     print(result)
