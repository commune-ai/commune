from web3.auto import w3
from eth_account import Account
from Crypto.Protocol.KDF import PBKDF2

# Prompt the user for a password and salt
password = input("Enter password: ")
salt = b'MySalt'

# Derive a key using PBKDF2
key = PBKDF2(password.encode(), salt, dkLen=32, count=100000)

# Create an account using the key
account = Account.privateKeyToAccount(key)

# Print the account address and private key
print("Account address:", account.address)
print("Private key:", account.privateKey.hex())
