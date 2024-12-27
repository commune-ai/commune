# ecdsa_helpers



Source: `commune/subspace/utils/ecdsa_helpers.py`

## Classes

### PrivateKey



#### Methods

##### `sign_msg(self, message: bytes) -> 'Signature'`



Type annotations:
```python
message: <class 'bytes'>
return: Signature
```

##### `sign_msg_hash(self, message_hash: bytes) -> 'Signature'`



Type annotations:
```python
message_hash: <class 'bytes'>
return: Signature
```

##### `sign_msg_hash_non_recoverable(self, message_hash: bytes) -> 'NonRecoverableSignature'`



Type annotations:
```python
message_hash: <class 'bytes'>
return: NonRecoverableSignature
```

##### `sign_msg_non_recoverable(self, message: bytes) -> 'NonRecoverableSignature'`



Type annotations:
```python
message: <class 'bytes'>
return: NonRecoverableSignature
```

##### `to_bytes(self) -> bytes`



Type annotations:
```python
return: <class 'bytes'>
```

##### `to_hex(self) -> str`



Type annotations:
```python
return: <class 'str'>
```

### PublicKey



#### Methods

##### `address(self)`



### Signature

Helper class that provides a standard way to create an ABC using
inheritance.

#### Methods

##### `recover_public_key_from_msg(self, message: bytes) -> eth_keys.datatypes.PublicKey`



Type annotations:
```python
message: <class 'bytes'>
return: <class 'eth_keys.datatypes.PublicKey'>
```

##### `recover_public_key_from_msg_hash(self, message_hash: bytes) -> eth_keys.datatypes.PublicKey`



Type annotations:
```python
message_hash: <class 'bytes'>
return: <class 'eth_keys.datatypes.PublicKey'>
```

##### `to_bytes(self) -> bytes`



Type annotations:
```python
return: <class 'bytes'>
```

##### `to_hex(self) -> str`



Type annotations:
```python
return: <class 'str'>
```

##### `to_non_recoverable_signature(self) -> 'NonRecoverableSignature'`



Type annotations:
```python
return: NonRecoverableSignature
```

##### `verify_msg(self, message: bytes, public_key: eth_keys.datatypes.PublicKey) -> bool`



Type annotations:
```python
message: <class 'bytes'>
public_key: <class 'eth_keys.datatypes.PublicKey'>
return: <class 'bool'>
```

##### `verify_msg_hash(self, message_hash: bytes, public_key: eth_keys.datatypes.PublicKey) -> bool`



Type annotations:
```python
message_hash: <class 'bytes'>
public_key: <class 'eth_keys.datatypes.PublicKey'>
return: <class 'bool'>
```

## Functions

### `bip39seed_to_bip32masternode(seed)`



### `derive_bip32childkey(parent_key, parent_chain_code, i)`



### `ecdsa_sign(private_key: bytes, message: bytes) -> bytes`



Type annotations:
```python
private_key: <class 'bytes'>
message: <class 'bytes'>
return: <class 'bytes'>
```

### `ecdsa_verify(signature: bytes, data: bytes, address: bytes) -> bool`



Type annotations:
```python
signature: <class 'bytes'>
data: <class 'bytes'>
address: <class 'bytes'>
return: <class 'bool'>
```

### `eth_utils_keccak(primitive: Union[bytes, int, bool, NoneType] = None, hexstr: Optional[str] = None, text: Optional[str] = None) -> bytes`



Type annotations:
```python
primitive: typing.Union[bytes, int, bool, NoneType]
hexstr: typing.Optional[str]
text: typing.Optional[str]
return: <class 'bytes'>
```

### `mnemonic_to_bip39seed(mnemonic, passphrase)`



### `mnemonic_to_ecdsa_private_key(mnemonic: str, str_derivation_path: str = None, passphrase: str = '') -> bytes`



Type annotations:
```python
mnemonic: <class 'str'>
str_derivation_path: <class 'str'>
passphrase: <class 'str'>
return: <class 'bytes'>
```

### `parse_derivation_path(str_derivation_path)`



### `to_checksum_address(value: Union[~AnyAddress, str, bytes]) -> eth_typing.evm.ChecksumAddress`

Makes a checksum address given a supported format.

Type annotations:
```python
value: typing.Union[~AnyAddress, str, bytes]
return: eth_typing.evm.ChecksumAddress
```

