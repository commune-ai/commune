# Commune Key Module

## Overview

The Key module provides a comprehensive cryptographic key management system for the Commune framework. It supports multiple cryptographic types (sr25519, ed25519, and ecdsa) and offers functionality for key generation, signing, verification, encryption, and decryption.

## Features

- **Multi-Crypto Support**: Compatible with sr25519 (Polkadot/Substrate), ed25519, and ECDSA (Ethereum/Bitcoin)
- **Key Generation**: Create keys from mnemonics, private keys, or URIs
- **Secure Storage**: Store keys with optional encryption
- **Signing & Verification**: Sign and verify data with different cryptographic schemes
- **Encryption & Decryption**: Encrypt and decrypt data using AES
- **Mnemonic Support**: Generate and validate BIP39 mnemonics
- **Network Compatibility**: Works with multiple blockchain networks (Polkadot, Ethereum, Bitcoin, etc.)

## Installation

The Key module is part of the Commune core and is installed with the main framework:

```bash
pip install commune
```

## Usage

### Basic Key Operations

```python
import commune as c

# Create a new key
key = c.key()

# Generate a new key with specific crypto type
sr25519_key = c.key(crypto_type='sr25519')  # Default
ed25519_key = c.key(crypto_type='ed25519')
ecdsa_key = c.key(crypto_type='ecdsa')  # Compatible with Ethereum

# Create key from mnemonic
mnemonic = "bottom drive obey lake curtain smoke basket hold race lonely fit walk"
key_from_mnemonic = c.key(mnemonic=mnemonic)

# Create key from private key
key_from_private = c.key(private_key="0x...")  # Hex string or bytes
```

### Key Management

```python
# Add a key to storage
key.add_key("my_key", crypto_type="sr25519")

# Get a key from storage
my_key = key.get_key("my_key")

# List all keys
all_keys = key.keys()

# Remove a key
key.rm_key("my_key")
```

### Signing and Verification

```python
# Sign data
data = "Hello, Commune!"
signature = key.sign(data)

# Verify signature
is_valid = key.verify(data, signature)
print(f"Signature valid: {is_valid}")

# Sign with different formats
hex_signature = key.sign(data, mode="hex")  # Returns hex string
json_signature = key.sign(data, mode="json")  # Returns dict with signature info
```

### Encryption and Decryption

```python
# Encrypt data
data = "Secret message"
encrypted = key.encrypt(data, password="my_password")

# Decrypt data
decrypted = key.decrypt(encrypted, password="my_password")
print(f"Decrypted: {decrypted}")

# Encrypt a key
key.encrypt_key("my_key", password="secure_password")

# Decrypt a key
key.decrypt_key("my_key", password="secure_password")
```

### Advanced Usage

```python
# Create a key from URI (with derivation path)
key_from_uri = c.key(suri="//Alice")

# Convert between formats
address = key.key_address  # SS58 address for sr25519/ed25519, checksum address for ecdsa
public_key = key.public_key  # Bytes
private_key = key.private_key  # Bytes

# Export key as JSON
key_json = key.to_json()

# Create key from JSON
key_from_json = c.key().from_json(key_json)
```

## Security Considerations

- Private keys should be kept secure and never shared
- Consider using encrypted storage for sensitive keys
- Use strong passwords for key encryption
- Backup your mnemonics in a secure location

## API Reference

### Key Class

- `__init__(private_key=None, mnemonic=None, crypto_type='sr25519', path=None, storage_path=None, **kwargs)`
- `set_private_key(private_key, crypto_type, mnemonic=None, **kwargs)`
- `get_crypto_type(crypto_type=None)`
- `add_key(path, crypto_type=None, mnemonic=None, refresh=False, private_key=None, **kwargs)`
- `get_key(path, password=None, create_if_not_exists=True, prompt_password=False, crypto_type=None, **kwargs)`
- `sign(data, mode='bytes')`
- `verify(data, signature=None, address=None, public_key=None, max_age=None, crypto_type=None, **kwargs)`
- `encrypt(data, password=None, key=None)`
- `decrypt(data, password=None, key=None)`
- `generate_mnemonic(words=24)`
- `from_mnemonic(mnemonic=None, crypto_type='sr25519')`
- `from_private_key(private_key, crypto_type='sr25519')`
- `from_uri(suri, crypto_type='sr25519')`

## License

This module is part of the Commune framework and is licensed under the same terms.
