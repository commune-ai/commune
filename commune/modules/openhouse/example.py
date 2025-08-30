#!/usr/bin/env python3
"""
Example script demonstrating the L2 Bitcoin rollup using state channels.
"""

import time
import json
import hashlib
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

from state_channel import StateChannel
from rollup_aggregator import RollupAggregator
from bitcoin_interface import BitcoinInterface
from rollup_node import RollupNode

def generate_key_pair():
    """Generate a new EC key pair for testing."""
    private_key = ec.generate_private_key(ec.SECP256K1())
    public_key = private_key.public_key()
    
    # Serialize the public key
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return {
        'private_key': private_key,
        'public_key': public_bytes.decode('utf-8')
    }

def main():
    print("L2 Bitcoin Rollup via State Channels - Example")
    print("-" * 50)
    
    # Generate key pairs for participants
    alice = generate_key_pair()
    bob = generate_key_pair()
    
    print("Created participants:")
    print(f"Alice: {alice['public_key'][:64]}...")
    print(f"Bob: {bob['public_key'][:64]}...")
    
    # Create a rollup node
    node = RollupNode(node_id="example_node", network="testnet")
    node.start()
    
    # Create a state channel
    channel_id = hashlib.sha256(f"channel:{int(time.time())}".encode()).hexdigest()
    participants = [alice['public_key'], bob['public_key']]
    initial_balances = {
        alice['public_key']: 5000000,  # 0.05 BTC in satoshis
        bob['public_key']: 3000000,    # 0.03 BTC in satoshis
    }
    
    print("\nCreating state channel...")
    result = node.create_channel(channel_id, participants, initial_balances)
    print(f"Channel created with ID: {channel_id}")
    print(f"Multisig address: {result['multisig_address']['address']}")
    
    # Fund the channel
    funding_tx = {
        'address': result['multisig_address']['address'],
        'amount': 0.08,  # 0.08 BTC
        'inputs': [{'dummy': 'input'}]
    }
    
    print("\nFunding the channel...")
    fund_result = node.fund_channel(channel_id, funding_tx)
    print(f"Channel funded with transaction: {fund_result['transaction']['txid']}")
    
    # Perform off-chain transactions
    print("\nPerforming off-chain transactions...")
    
    # Transaction 1: Alice sends 0.01 BTC to Bob
    new_balances_1 = {
        alice['public_key']: 4000000,  # 0.04 BTC
        bob['public_key']: 4000000,    # 0.04 BTC
    }
    
    # In a real implementation, both parties would sign the state
    # For simulation, we'll create dummy signatures
    signatures_1 = {
        alice['public_key']: "alice_signature_1",
        bob['public_key']: "bob_signature_1"
    }
    
    update_result_1 = node.update_channel(channel_id, new_balances_1, signatures_1)
    print(f"Transaction 1 complete. New state nonce: {update_result_1['channel_state']['nonce']}")
    
    # Transaction 2: Bob sends 0.005 BTC back to Alice
    new_balances_2 = {
        alice['public_key']: 4500000,  # 0.045 BTC
        bob['public_key']: 3500000,    # 0.035 BTC
    }
    
    signatures_2 = {
        alice['public_key']: "alice_signature_2",
        bob['public_key']: "bob_signature_2"
    }
    
    update_result_2 = node.update_channel(channel_id, new_balances_2, signatures_2)
    print(f"Transaction 2 complete. New state nonce: {update_result_2['channel_state']['nonce']}")
    
    # Simulate batch processing
    print("\nProcessing transaction batch...")
    time.sleep(2)  # Wait for batch processing
    
    # Initiate channel closing
    print("\nInitiating channel close...")
    close_init = node.initiate_channel_close(channel_id)
    print(f"Channel closing initiated. Dispute period ends at: {close_init['dispute_period_ends']}")
    
    # Simulate dispute period passing
    print("\nSimulating dispute period...")
    # In a real implementation, we would wait for the dispute period
    # For simulation, we'll modify the closing time directly
    channel = node.channels[channel_id]
    channel.closing_time = int(time.time()) - channel.dispute_period - 1
    
    # Close the channel
    print("\nClosing the channel...")
    close_result = node.close_channel(channel_id)
    print(f"Channel closed with transaction: {close_result['transaction']['txid']}")
    print("Final balances:")
    for participant, balance in close_result['final_balances'].items():
        print(f"  {participant[:16]}...: {balance/100000000} BTC")
    
    # Stop the node
    node.stop()
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
