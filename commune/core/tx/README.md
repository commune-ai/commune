# Commune Transaction Module

This module provides transaction handling capabilities for Commune, allowing for:

- Recording function calls with their inputs and outputs
- Signing transactions with cryptographic keys
- Storing and retrieving transaction history
- Verifying transaction integrity

## Usage

```python
import commune as c

# Initialize the transaction module
tx = c.module('tx')()

# View transaction history
tx.txs()

# Test the transaction module
tx.test()
```

## Features

- Transaction signing and verification
- Persistent storage of transaction history
- Serialization of complex objects
- Query capabilities for transaction history
