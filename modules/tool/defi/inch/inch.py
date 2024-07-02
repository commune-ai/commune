import requests 
import json
from typing import Optional, Dict, Any
import time

import os
from dotenv import load_dotenv
from web3 import Web3
import commune as c
from dotenv import load_dotenv


class Inch(c.Module):
    def __init__(self, 
                 api_key: Optional[str] = 'ONEINCH_API_KEY',
                url: Optional[str] = "https://api.1inch.dev/price/v1.1/1"
                 ):
        self.api_key = os.getenv(api_key, api_key)
        self.url = url



    def get_whitelisted_token_prices(self):
        
        response = requests.get(self.url,  headers={'Authorization': f'Bearer {self.api_key}'})
        if response.status_code == 200:
            prices = response.json()
            print("Prices for whitelisted tokens:")
            token2price = {}
            for token_address, price in prices.items():
                print(f"{token_address}: {price}")
                token2price[token_address] = price
        else:
            print("Failed to fetch token prices.")
            

    def get_requested_token_prices(self, tokens:list[str]):

        payload = {
            "tokens": tokens
        }

        response = requests.post(self.url, json=payload, headers={'Authorization': f'Bearer {self.API_KEY}'})
        if response.status_code == 200:
            prices = response.json()
            print("Prices for requested tokens:")
            for token_address, price in prices.items():
                print(f"{token_address}: {price}")
            return (f"{token_address}: {price}")
        else:
            print("Failed to fetch token prices.")
        

    def get_prices_for_addresses(self,addresses):
        url = f"{self.url}{','.join(addresses)}"

        response = requests.get(url,  headers={'Authorization': f'Bearer {self.API_KEY}'})
        if response.status_code == 200:
            prices = response.json()
            print("Prices for requested tokens:")
            response = {}
            for token_address, price in prices.items():
                response[token_address] = price
        else:
            response =  {"error": f"Failed to fetch token prices. Error code: {response.status_code}"}

        return response
    
        
    
