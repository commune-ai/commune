# Module Registry Pallet

A Substrate pallet for managing a decentralized module registry with multi-chain support.

## Overview

This pallet provides a simple key-value storage system where:
- **Keys**: Public keys in various formats (Ed25519, Ethereum, Solana) stored as `Vec<u8>`
- **Values**: IPFS CIDs pointing to module metadata stored as `Vec<u8>`

## Features

- **Multi-chain compatibility**: Support for Ed25519, Ethereum, and Solana public keys
- **Reduced on-chain storage costs**: Only CIDs stored on-chain, full metadata on IPFS
- **Content-addressable metadata**: Via IPFS integration
- **Flexible key format**: No schema constraints, supports various public key formats
- **Chain-agnostic design**: Easy integration across different blockchain networks

## Storage Design

The core storage is a simple `StorageMap`:
```rust
StorageMap<_, Blake2_128Concat, BoundedVec<u8, MaxKeyLength>, BoundedVec<u8, MaxCidLength>, OptionQuery>
```

- **Key**: `BoundedVec<u8, MaxKeyLength>` - Public key in various formats
- **Value**: `BoundedVec<u8, MaxCidLength>` - IPFS CID pointing to module metadata

## Dispatchable Functions

### `register_module(key: Vec<u8>, cid: Vec<u8>)`
Register a new module in the registry by storing an IPFS CID for a given public key.

### `update_module(key: Vec<u8>, cid: Vec<u8>)`
Update an existing module's IPFS CID.

### `remove_module(key: Vec<u8>)`
Remove a module from the registry.

## Events

- `ModuleRegistered`: Emitted when a module is successfully registered
- `ModuleUpdated`: Emitted when a module is successfully updated
- `ModuleRemoved`: Emitted when a module is successfully removed

## Errors

- `ModuleNotFound`: The module does not exist in the registry
- `InvalidKeyFormat`: The public key format is invalid
- `InvalidCidFormat`: The IPFS CID format is invalid
- `KeyTooLong`: The public key exceeds maximum length
- `CidTooLong`: The IPFS CID exceeds maximum length
- `EmptyKey`: The public key is empty
- `EmptyCid`: The IPFS CID is empty
- `ModuleAlreadyExists`: The module already exists in the registry

## Configuration

The pallet requires the following configuration parameters:

```rust
type MaxKeyLength: Get<u32>;  // Maximum length for public keys (recommended: 128)
type MaxCidLength: Get<u32>;  // Maximum length for IPFS CIDs (recommended: 128)
```

## Supported Key Formats

The pallet validates and supports various public key formats:
- **Ed25519**: 32 bytes
- **Ethereum Address**: 20 bytes
- **Ethereum Public Key**: 64 bytes
- **Solana**: 32 bytes
- **Bitcoin P2PKH**: 20 bytes
- **Bitcoin P2WSH**: 32 bytes
- **Custom formats**: 16-128 bytes (flexible for future blockchain support)

## IPFS CID Validation

The pallet performs basic validation on IPFS CIDs:
- Supports both CIDv0 and CIDv1 formats
- Length validation (32-128 characters)
- Character validation (alphanumeric + `-` and `_`)
- UTF-8 encoding validation

## Usage Example

```rust
// Register a module
let ed25519_key = vec![0u8; 32];  // 32-byte Ed25519 public key
let ipfs_cid = b"QmTestCID123456789012345678901234".to_vec();

ModuleRegistry::register_module(
    RuntimeOrigin::signed(account_id),
    ed25519_key,
    ipfs_cid
)?;

// Update a module
let new_cid = b"QmNewCID1234567890123456789012345".to_vec();
ModuleRegistry::update_module(
    RuntimeOrigin::signed(account_id),
    ed25519_key.clone(),
    new_cid
)?;

// Remove a module
ModuleRegistry::remove_module(
    RuntimeOrigin::signed(account_id),
    ed25519_key
)?;
```

## Integration with IPFS

This pallet is designed to work with the `commune-ipfs` submodule for off-chain metadata storage:

1. **Store metadata on IPFS**: Use the IPFS API to store module metadata and get a CID
2. **Register on-chain**: Use this pallet to store the CID on-chain with the public key
3. **Retrieve metadata**: Use the CID to fetch metadata from IPFS when needed

## Testing

Run the test suite:
```bash
cargo test -p pallet-module-registry
```

## Benchmarking

Run benchmarks:
```bash
cargo run --release --features runtime-benchmarks -- benchmark pallet --pallet pallet_module_registry --extrinsic "*"
```
