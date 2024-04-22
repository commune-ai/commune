from web3 import Web3 
import os
#connect to a local ethereum node
w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

#AAVE ADDRESS 
contract_address = '0x6Ae43d3271ff6888e7Fc43Fd7321a503ff738951'
contract_abi = []

contract = w3.eth.contract(address=contract_address, abi=contract_abi)

def allowance(owner_address, spender_address, token_address):
    # Get the ERC20 Token contract
    erc20_contract = w3.eth.contract(address=token_address, abi=contract_abi)
    
    # Call the allowance function
    allowed_amount = erc20_contract.functions.allowance(owner_address, spender_address).call()
    return allowed_amount

def supply(dst_address, asset_address, amount):
    # Replace with the address that will send the transaction
    sender_address = '0xYourSenderAddress'
    
    # Build the transaction
    transaction = contract.functions.supplyTo(dst_address, asset_address, amount).buildTransaction({
        'from': sender_address,
        'gas': 2000000,
        'gasPrice': w3.toWei('20', 'gwei'),
        'nonce': w3.eth.getTransactionCount(sender_address),
    })
    
    # Sign the transaction
    PRIVATE_KEY= os.getenv("PRIVATE_KEY", "")
    signed_transaction = w3.eth.account.signTransaction(transaction, PRIVATE_KEY)
    
    # Send the transaction
    tx_hash = w3.eth.sendRawTransaction(signed_transaction.rawTransaction)
    return tx_hash.hex()
