import time
from web3 import Web3
from typing import Optional

class Access:
    def __init__(self, contract_address: str, abi: list, provider_url: str, decimals: int = 18):
        self.contract_address = contract_address
        self.abi = abi
        self.provider_url = provider_url
        self.decimals = decimals
        self.web3 = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)
        self.last_request_time = {}

    def get_token_balance(self, address: str) -> int:
        balance = self.contract.functions.balanceOf(address).call()
        return balance // (10 ** self.decimals)

    def get_rate_limit(self, address: str) -> int:
        token_balance = self.get_token_balance(address)
        if token_balance == 0:
            return 0
        return token_balance * 10  # Adjust the multiplier as needed

    def check_rate_limit(self, address: str) -> bool:
        current_time = time.time()
        if address not in self.last_request_time:
            self.last_request_time[address] = current_time
            return True

        rate_limit = self.get_rate_limit(address)
        if rate_limit == 0:
            return False

        time_diff = current_time - self.last_request_time[address]
        if time_diff < (1 / rate_limit):
            return False

        self.last_request_time[address] = current_time
        return True

    def make_request(self, address: str, func, *args, **kwargs) -> Optional[any]:
        if not self.check_rate_limit(address):
            print(f"Rate limit exceeded for address: {address}")
            return None

        return func(*args, **kwargs)