import commune 
import streamlit as st

# commune.get_module("web3.account.substrate").from_mnemonic("")


from substrateinterface import Keypair

# Generate a substrate keypair from a string
my_string = "Hello, world!"
keypair = Keypair.create_from_mnemonic(my_string)

# Print the generated keypair
print(f"Public key: {keypair.public_key}")
print(f"Secret key: {keypair.secret_key}")