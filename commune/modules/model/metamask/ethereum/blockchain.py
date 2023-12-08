from web3 import Web3

def process_transaction(transaction_data):
    web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/a7b2e24f3c5640bd9c588f02c3039e6d'))

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
        'chainId': 1  # 1 for Ethereum mainnet
    }

    # Signing the transaction
    signed_tx = web3.eth.account.sign_transaction(transaction, private_key)

    # Sending the transaction
    tx_hash = web3.eth.sendRawTransaction(signed_tx.rawTransaction)

    return {"status": "success", "transaction_hash": tx_hash.hex()}