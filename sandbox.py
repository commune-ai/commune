# Setup
from web3 import Web3

alchemy_url = "https://eth-mainnet.g.alchemy.com/v2/RrtpZjiUVoViiDEaYxhN9o6m1CSIZvlL"
w3 = Web3(Web3.HTTPProvider(alchemy_url))

# Print if web3 is successfully connected
print(w3.isConnected())

# Get the latest block number
latest_block = w3.eth.block_number
print(latest_block)

# Get the balance of an account
balance = w3.eth.get_balance('0x742d35Cc6634C0532925a3b844Bc454e4438f44e')
print(balance)

# Get the information of a transaction
tx = w3.eth.get_transaction('0x5c504ed432cb51138bcf09aa5e8a410dd4a1e204ef84bfed1be16dfba1b22060')
print(tx)