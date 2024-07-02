from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware
import requests
import os
import time

# Environment variables for sensitive data
INFURA_URL = os.getenv('INFURA_URL', 'https://mainnet.infura.io/v3/a7b2e24f3c5640bd9c588f02c3039e6d')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY', 'FU6MK1CV1RCIV824CWBXJQ7UNPD9DTCTTZ')

# Initialize Web3
web3 = Web3(HTTPProvider(INFURA_URL))
web3.middleware_onion.inject(geth_poa_middleware, layer=0)

def process_transaction(transaction_data):
    try:
        from_address = transaction_data['from']
        to_address = transaction_data['to']
        value = web3.toWei(transaction_data['value'], 'ether')  
        private_key = transaction_data['private_key']

        nonce = web3.eth.getTransactionCount(from_address)
        gas_estimate = web3.eth.estimateGas({'to': to_address, 'from': from_address, 'value': value})
        gas_price = web3.eth.gasPrice

        transaction = {
            'to': to_address,
            'value': value,
            'gas': gas_estimate,
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': 1  # Ethereum mainnet
        }

        # Signing the transaction
        signed_tx = web3.eth.account.sign_transaction(transaction, private_key)

        # Sending the transaction
        tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)

        # Wait for transaction receipt (confirmation)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return {"status": "success", "transaction_hash": tx_hash.hex(), "receipt": receipt}

    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_transaction_history(address):
    url = f'https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            transactions = response.json()['result']
            return {"status": "success", "transactions": transactions}
        else:
            return {"status": "error", "message": "Failed to fetch transaction history"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def make_uniswap_trade(trade_data):
    return {"status": "info", "message": "Uniswap trade functionality not implemented yet"}