# Arweave Storage Module

Permanent decentralized storage using Arweave blockchain.

## Features

- Permanent data storage
- Decentralized architecture
- Transaction-based retrieval
- Metadata tagging support

## Usage

```python
from commune.modules.store.arweave import ArweaveStore

# Initialize store
store = ArweaveStore(gateway="https://arweave.net")

# Store data
tx_id = store.store(data={"key": "value"}, tags={"type": "json"})

# Retrieve data
data = store.retrieve(tx_id)

# Check status
status = store.get_status(tx_id)
```

## Requirements

- requests
- Valid Arweave wallet (for uploads)

## License

MIT