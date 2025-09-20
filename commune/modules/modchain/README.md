# Substrate Module Registry with Local Testnet

A Substrate blockchain with a custom module registry pallet that supports SR25519 and ECDSA keys.

## Features

- **Module Registry Pallet**: Register modules with public keys as identifiers
- **Multiple Key Types**: Support for SR25519 and ECDSA
- **JSON Storage**: Store module metadata as JSON
- **Ownership Model**: Only owners can update/remove entries
- **Signature Verification**: Built-in signature verification
- **Local Testnet**: Easy-to-run local development network

## Quick Start

### Prerequisites

- Docker and docker-compose (for testnet)
- Rust and Cargo (for building from source)

### Running Local Testnet

1. Start the testnet:
```bash
./scripts/run-testnet.sh
```

2. Access the nodes:
- Alice WebSocket: ws://localhost:9944
- Alice RPC: http://localhost:9933
- Bob WebSocket: ws://localhost:9945
- Bob RPC: http://localhost:9934

3. Stop the testnet:
```bash
docker-compose down
```

### Building from Source

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Build the node:
```bash
./scripts/build.sh
```

3. Run a development node:
```bash
./target/release/substrate-node --dev
```

## Using the Module Registry

### Register a Module

```javascript
// Using Polkadot.js API
const keyType = 'Sr25519';
const keyData = '0x' + publicKey; // 32 bytes for SR25519
const valueType = 'Json';
const valueData = JSON.stringify({
  name: 'my_module',
  version: '1.0.0',
  description: 'My awesome module'
});

await api.tx.moduleRegistry
  .register(keyType, keyData, valueType, valueData)
  .signAndSend(alice);
```

### Query a Module

```javascript
const key = {
  keyType: 'Sr25519',
  data: keyData
};

const module = await api.query.moduleRegistry.registry(key);
console.log(module.toJSON());
```

### Update a Module

```javascript
await api.tx.moduleRegistry
  .update(keyType, keyData, valueType, newValueData)
  .signAndSend(alice);
```

### Verify Signature

```javascript
const message = 'Hello, Substrate!';
const signature = sign(message, privateKey);

await api.tx.moduleRegistry
  .verifySignature(keyType, keyData, message, signature)
  .signAndSend(alice);
```

## Architecture

### Pallet Structure

- **Storage**: Single map with `RegistryKey -> RegistryValue`
- **Key Types**: SR25519 (32 bytes), ECDSA (33 bytes compressed)
- **Value Types**: JSON, Bytes
- **Events**: Registered, Updated, Unregistered, Verified

### Docker Setup

- Two nodes: Alice and Bob
- Automatic peer discovery
- Persistent volumes for blockchain data
- Exposed WebSocket and RPC endpoints

## Development

### Running Tests

```bash
cargo test
```

### Viewing Logs

```bash
# Docker logs
docker-compose logs -f

# Native node logs
./target/release/substrate-node --dev -l debug
```

### Connecting Frontend

Use [Polkadot.js Apps](https://polkadot.js.org/apps) and connect to:
- Local node: ws://localhost:9944

## Troubleshooting

### Port Already in Use

If ports are already in use, modify the port mappings in `docker-compose.yml`.

### Build Errors

Ensure you have the latest Rust version:
```bash
rustup update
rustup target add wasm32-unknown-unknown
```

### Docker Issues

Reset the testnet:
```bash
docker-compose down -v
docker-compose up -d
```

## License

Apache-2.0