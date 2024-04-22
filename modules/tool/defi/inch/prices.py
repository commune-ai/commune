import requests 
import json
from typing import Optional, Dict, Any
import time
from typing import *

import os
from dotenv import load_dotenv
from web3 import Web3
import commune as c

from dotenv import load_dotenv

class InchPrices(c.Module):
    description = """

    Gets the token prices from the 1inch API.

    params: A list of token addresses.
    return: A dictionary of token addresses and prices.

    """
    def __init__(self, 
                 api_key: Optional[str] = 'INCH_API_KEY',
                url: Optional[str] = "https://api.1inch.dev/price/v1.1/1"
                 ):
        self.api_key = os.getenv(api_key, api_key)
        self.url = url

        self.token_mappings = {
        "usdc": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "wsteth": "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",
        "reth": "0xae78736Cd615f374D3085123A210448E74Fc6393",
        "dai": "0x6b175474e89094c44da98b954eedeac495271d0f",
        "usdt": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "wbtc": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
        "weth": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    }
        self.reverse_token_mappings = {v: k for k, v in self.token_mappings.items()}


    def call(self, tokens: List[str]= ['weth']) -> dict[str, float]:

        for i,t in enumerate(tokens):
            if t.lower() in self.token_mappings:
                tokens[i] = self.token_mappings.get(t.lower())           

        payload = {
            "tokens": tokens
        }

        response = requests.post(self.url, json=payload, headers={'Authorization': f'Bearer {self.api_key}'})
        if response.status_code == 200:
            prices = response.json()
            print("Prices for requested tokens:")
            response = {}
            for token_address, price in prices.items():
                if token_address in self.reverse_token_mappings:
                    response[self.reverse_token_mappings[token_address]] = price
                else:
                    response[token_address] = price
                
        else:
            print("Failed to fetch token prices.", response.text)

        return response
    

if __name__ == "__main__":
    instance = InchPrices()
    result = instance.call(tokens=['weth', 'usdc'])
    print(result)
#      aave_instance = AaveV3()
#      result=aave_instance.call(chain="Ethereum", symbol="WETH")
#      print(result)
