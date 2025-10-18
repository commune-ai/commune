# Decentralized Storage Store

File system compatible implementations for Arweave, IPFS, and Filecoin.

## Backends

### Arweave
Permanent, immutable storage on the Arweave network.

```python
from store import ArweaveStore

store = ArweaveStore(gateway="https://arweave.net")
tx_id = store.write(b"data", tags=[{"name": "Content-Type", "value": "text/plain"}])
data = store.read(tx_id)
```

### IPFS
Content-addressed, peer-to-peer storage.

```python
from store import IPFSStore

store = IPFSStore(host="127.0.0.1", port=5001)
cid = store.write(b"data")
data = store.read(cid)
store.pin(cid)  # Prevent garbage collection
```

### Filecoin
Decentralized storage marketplace with provable storage.

```python
from store import FilecoinStore

store = FilecoinStore(api_url="https://api.node.glif.io")
deal = store.create_deal(data_cid, miner, price, duration)
status = store.get_deal_status(deal_cid)
```

### Unified Interface

```python
from store import UnifiedStore

# Use IPFS
store = UnifiedStore('ipfs', host='127.0.0.1')
id = store.write(b"data")
data = store.read(id)

# Switch to Arweave
store = UnifiedStore('arweave')
tx_id = store.write(b"data")
```

## Features

- **File system compatible**: Read/write operations similar to local filesystem
- **Multiple backends**: Arweave, IPFS, Filecoin support
- **Unified interface**: Switch backends without changing code
- **Metadata support**: Tags, pins, deal management
- **Gateway fallbacks**: Public gateways for read operations

## Installation

```bash
pip install requests
# For full functionality:
pip install arweave-python-client ipfshttpclient
```
