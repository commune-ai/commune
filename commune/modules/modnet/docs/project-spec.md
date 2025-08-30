Absolutely. Here's a clear and tight **spec document** for your CID-based on-chain module registry. This spec assumes the use of Substrate, IPFS, and a Python client, and is designed for easy iteration while preserving forward compatibility.

---

# 🛠️ Module Registry Specification (v0.1)

## Overview

This registry stores and retrieves metadata for computational modules using a simplified architecture. All metadata is stored off-chain in IPFS, and the on-chain registry maps multi-chain public keys to IPFS CIDs only.

The goal is to provide a minimal, scalable, and content-addressable module registry with multi-chain compatibility, suitable for evolving systems and decentralized infrastructure.

---

## ✅ Requirements

* [x] Allow arbitrary modules to register metadata via a simple CRUD interface
* [x] Store IPFS CIDs as `Vec<u8>` values (no full metadata on-chain)
* [x] Accept multi-chain public keys as `Vec<u8>` keys (Ed25519, Ethereum, Solana)
* [x] Be chain-agnostic and easily pluggable into any Substrate runtime
* [x] Python client compatibility with multi-chain key encoding
* [x] Reduced on-chain storage costs through CID-only storage
* [x] Content-addressable metadata via IPFS integration

---

## 🔗 IPFS Integration

The module uses `commune-ipfs` as a submodule for distributed storage:

```bash
# Initialize submodule
git submodule add https://github.com/bakobiibizo/commune-ipfs
```

Features provided by IPFS integration:
- 🚀 FastAPI backend for file management
- 📁 Distributed storage with content addressing
- 🔍 Metadata search and filtering
- 📊 Local SQLite for file indexing
- 🌐 Web interface for file management

IPFS Configuration:
```bash
# Default endpoints
IPFS_API_URL=http://localhost:5001
IPFS_GATEWAY_URL=http://localhost:8080
```

## 🧱 On-Chain Storage

```rust
#[pallet::storage]
#[pallet::getter(fn module_registry)]
pub(super) type ModuleRegistry<T: Config> = StorageMap<
    _,                 // Default: Blake2_128Concat
    Blake2_128Concat,  // Hasher for efficient key lookups
    Vec<u8>,           // Key: Multi-chain public key (Ed25519, Ethereum, Solana)
    Vec<u8>,           // Value: IPFS CID pointing to off-chain metadata
>;
```

### Multi-Chain Key Support

- **Ed25519**: 32-byte public keys (Polkadot/Substrate native)
- **Ethereum**: 20-byte addresses or 64-byte public keys
- **Solana**: 32-byte public keys
- **Flexible**: `Vec<u8>` format supports future key types
```

---

## 🔁 Off-Chain Metadata Schema

The **recommended** metadata structure (stored in IPFS or directly on-chain):

```json
{
  "name": "text-gen-v1",
  "version": "1.0.0",
  "author": "Bakobiibizo",
  "cid": "bafybeibwxyzmoduledata",
  "entrypoint": "main.py",
  "args": {
    "max_tokens": 512,
    "temperature": 0.7
  },
  "custom": {
    "gpu": "A6000",
    "runtime": "python3.10"
  }
}
```

Notes:

* `cid` points to containerized code or resource blob
* `custom` allows freeform extension without schema migration
* Stored either:

  * In IPFS, with only the `cid` stored on-chain
  * Entire JSON blob stored directly on-chain (small modules)

---

## 🧪 Runtime Calls

### `register_module(key: Vec<u8>, cid: Vec<u8>)`

* Inserts or updates a module entry
* `key`: Multi-chain public key (Ed25519, Ethereum, Solana)
* `cid`: IPFS CID pointing to off-chain metadata
* Validates key format and CID structure

### `get_module(key: Vec<u8>) -> Option<Vec<u8>>`

* Returns the stored IPFS CID for the given public key
* Use CID to fetch metadata from IPFS

### `remove_module(key: Vec<u8>)`

* Deletes a module entry from the registry
* Does not remove metadata from IPFS (content-addressed)

---

## 🔒 Constraints

* **Key constraints**:
  - Ed25519: 32 bytes
  - Ethereum addresses: 20 bytes
  - Ethereum public keys: 64 bytes
  - Solana: 32 bytes
  - Max key length: 64 bytes (enforced by runtime)
* **Value constraints**:
  - IPFS CID format validation
  - Typical CID length: 34-59 bytes
  - Max value length: 128 bytes (CID storage only)
* **Uniqueness**: Keys must be unique per public key owner
* **Validation**: Both key format and CID structure validated

---

## 🧠 Design Considerations

* **Content-addressed**: IPFS ensures immutability of metadata
* **Multi-chain compatibility**: Support for Ed25519, Ethereum, and Solana keys
* **Cost-efficient**: Only CIDs stored on-chain, reducing storage costs
* **Decoupled storage**: Metadata lives off-chain, registry provides addressing
* **Extensible**: Flexible key format supports future blockchain integrations
* **Composable**: Easily layered with future consensus registry or version control
* **Immutable references**: CIDs provide tamper-proof metadata addressing

---

## 🐍 Python Client Example

```python
from substrateinterface import SubstrateInterface
import ipfshttpclient
from cryptography.hazmat.primitives import serialization

# Initialize connections
substrate = SubstrateInterface(
    url="ws://127.0.0.1:9944",
    type_registry_preset='substrate-node-template'
)
ipfs = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')

# Prepare metadata
metadata = {
    "name": "text-gen-v1",
    "version": "1.0.0",
    "author": "Bakobiibizo",
    "entrypoint": "main.py",
    "args": {"max_tokens": 512, "temperature": 0.7},
    "custom": {"gpu": "A6000", "runtime": "python3.10"}
}

# Upload metadata to IPFS
result = ipfs.add_json(metadata)
cid = result['Hash'].encode('utf-8')

# Use public key as registry key (example: Ed25519)
public_key = b"\x12\x34..."  # 32-byte Ed25519 public key

# Register module with CID
substrate.compose_call(
    call_module='ModuleRegistry',
    call_function='register_module',
    call_params={'key': public_key, 'cid': cid}
)

# Retrieve and fetch metadata
stored_cid = substrate.query(
    module='ModuleRegistry',
    storage_function='module_registry',
    params=[public_key]
)
if stored_cid:
    metadata = ipfs.get_json(stored_cid.decode('utf-8'))
```

---

## 🧪 Testing Plan

* [ ] Unit tests: insert, update, delete, and retrieve raw values
* [ ] Round-trip tests: serialize JSON → insert → fetch → parse
* [ ] IPFS mock: CID fetch + JSON integrity test
* [ ] Fuzz tests for large `Vec<u8>` edge cases

---

## 🔮 Future Upgrades

* Add per-module permissions and signatures
* Add indexing by author, tags, or version
* Extend registry to support consensus-weighted scoring
* Add optional IPFS pinning or replication hooks
* Implement IPFS cluster for redundant storage
* Add automatic garbage collection for unused modules
* Integrate IPFS peer discovery for module sharing
