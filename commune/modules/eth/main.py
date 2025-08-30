# Replace with your own provider (e.g., Infura or Ganache)

from web3 import Web3
from solcx import compile_source
import json
import time


w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))

# Replace with your own private keys and addresses
private_key_A = '0x...'
private_key_B = '0x...'
account_A = w3.eth.account.from_key(private_key_A)
account_B = w3.eth.account.from_key(private_key_B)

# Solidity source code
contract_source = '''
// [Insert the Solidity contract code here]
'''

# Compile the contract
compiled_sol = compile_source(contract_source)
contract_interface = compiled_sol['<stdin>:CallCounter']

# Deploy the contract
def deploy_contract():
    # Get bytecode and abi
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']

    # Create contract in Python
    CallCounter = w3.eth.contract(abi=abi, bytecode=bytecode)

    # Build constructor transaction
    construct_txn = CallCounter.constructor(300).build_transaction({
        'from': account_A.address,
        'nonce': w3.eth.get_transaction_count(account_A.address),
        'gas': 6721975,
        'gasPrice': w3.toWei('10', 'gwei')
    })

    # Sign and send the transaction
    signed = w3.eth.account.sign_transaction(construct_txn, private_key_A)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    print(f'Deploying contract... Transaction hash: {tx_hash.hex()}')

    # Wait for the transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f'Contract deployed at address: {tx_receipt.contractAddress}')

    return tx_receipt.contractAddress, abi

# Function to interact with the contract
def update_call_count(contract_address, abi):
    # Create contract instance
    contract = w3.eth.contract(address=contract_address, abi=abi)

    # Create JSON data with timestamp
    timestamp = int(time.time())
    json_data = json.dumps({"timestamp": timestamp})

    # Sign the JSON data with both accounts
    message_hash = w3.keccak(text=json_data)
    eth_signed_message_hash = w3.solidityKeccak(
        ['string'], [f"\x19Ethereum Signed Message:\n32{message_hash.hex()}"]
    )

    signature_A = w3.eth.account.sign_message(
        {'messageHash': eth_signed_message_hash},
        private_key=private_key_A
    ).signature

    signature_B = w3.eth.account.sign_message(
        {'messageHash': eth_signed_message_hash},
        private_key=private_key_B
    ).signature

    # Build transaction to update call count
    txn = contract.functions.updateCallCount(
        json_data,
        signature_A,
        signature_B
    ).build_transaction({
        'from': account_A.address,
        'nonce': w3.eth.get_transaction_count(account_A.address),
        'gas': 500000,
        'gasPrice': w3.toWei('10', 'gwei')
    })

    # Sign and send the transaction
    signed_txn = w3.eth.account.sign_transaction(txn, private_key_A)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    print(f'Updating call count... Transaction hash: {tx_hash.hex()}')

    # Wait for the transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print('Call count updated.')

    # Fetch the updated call count
    count = contract.functions.callCounts(account_A.address, account_B.address).call()
    print(f'Call count from {account_A.address} to {account_B.address}: {count}')

# Main execution
if __name__ == '__main__':
    contract_address, abi = deploy_contract()
    update_call_count(contract_address, abi)
