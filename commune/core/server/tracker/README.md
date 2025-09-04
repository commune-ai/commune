# Commune Transaction (TX) Module

## Overview

The Transaction (TX) mod in Commune provides a standardized interface for handling blockchain transactions across different networks. It serves as a crucial component for interacting with various blockchain protocols, enabling seamless transaction creation, signing, and broadcasting.

## Features

- **Cross-Chain Compatibility**: Support for multiple blockchain networks
- **Transaction Management**: Create, sign, and broadcast transactions
- **Gas Estimation**: Automatic calculation of gas fees
- **Transaction Status Tracking**: Monitor transaction confirmations
- **Error Handling**: Robust error management for failed transactions

## Installation

The TX mod is included as part of the Commune core. No separate installation is required if you have Commune installed.

```bash
# If you need to install Commune
git clone https://github.com/commune-ai/commune.git
cd commune
make install
```

## Usage

```python
from commune.core.tx import TX

# Initialize a transaction handler
tx = TX(network='ethereum')

# Create and send a transaction
transaction = tx.transfer(
    to_address='0x...',
    amount=0.1,
    token='ETH'
)

# Get transaction status
status = tx.get_transaction_status(transaction.hash)
```

## Configuration

The TX mod can be configured through the Commune config system or by passing parameters during initialization:

```python
tx = TX(
    network='ethereum',
    provider='https://mainnet.infura.io/v3/YOUR_API_KEY',
    private_key='YOUR_PRIVATE_KEY',  # Optional, for signing transactions
    gas_limit=21000,                 # Optional, default gas limit
    gas_price_strategy='medium'      # Optional, gas price strategy
)
```

## API Reference

- `TX.transfer(to_address, amount, token)`: Send tokens to an address
- `TX.sign_transaction(transaction)`: Sign a transaction with the configured private key
- `TX.broadcast_transaction(signed_tx)`: Broadcast a signed transaction to the network
- `TX.get_transaction_status(tx_hash)`: Check the status of a transaction
- `TX.estimate_gas(transaction)`: Estimate the gas required for a transaction

## Contributing

Contributions to improve the TX mod are welcome. Please follow the standard Commune contribution guidelines.

## License

This mod is part of the Commune project and is released under the same license terms as the main project.
