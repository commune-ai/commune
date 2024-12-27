# key



Source: `commune/key.py`

## Classes

### Bytes

A variable collection of bytes, stored as an `Vec<u8>`

#### Methods

##### `decode(self, data: scalecodec.base.ScaleBytes = None, check_remaining=True)`

Decodes available SCALE-bytes according to type specification of this ScaleType

If no `data` is provided, it will try to decode data specified during init

If `check_remaining` is enabled, an exception will be raised when data is remaining after decoding

Parameters
----------
data
check_remaining: If enabled, an exception will be raised when data is remaining after decoding

Returns
-------

Type annotations:
```python
data: <class 'scalecodec.base.ScaleBytes'>
```

##### `encode(self, value=None) -> scalecodec.base.ScaleBytes`

Encodes the serialized `value` representation of current `ScaleType` to a `ScaleBytes` stream

Parameters
----------
value

Returns
-------
ScaleBytes

Type annotations:
```python
return: <class 'scalecodec.base.ScaleBytes'>
```

##### `get_next_bool(self) -> bool`

Retrieves the next byte and convert to an bool

Returns
-------
bool

Type annotations:
```python
return: <class 'bool'>
```

##### `get_next_bytes(self, length) -> bytearray`

Retrieve `length` amount of bytes of the SCALE-bytes stream

Parameters
----------
length: amount of requested bytes

Returns
-------
bytearray

Type annotations:
```python
return: <class 'bytearray'>
```

##### `get_next_u8(self) -> int`

Retrieves the next byte and convert to an int

Returns
-------
int

Type annotations:
```python
return: <class 'int'>
```

##### `get_remaining_bytes(self) -> bytearray`

Retrieves all remaining bytes from the stream

Returns
-------
bytearray

Type annotations:
```python
return: <class 'bytearray'>
```

##### `get_used_bytes(self) -> bytearray`

Returns a bytearray of all SCALE-bytes used in the decoding process

Returns
-------
bytearray

Type annotations:
```python
return: <class 'bytearray'>
```

##### `is_primitive(type_string: str) -> bool`



Type annotations:
```python
type_string: <class 'str'>
return: <class 'bool'>
```

##### `process(self)`

Implementation of the decoding process

Returns
-------

##### `process_encode(self, value)`

Implementation of the encoding process

Parameters
----------
value

Returns
-------
ScaleBytes

##### `process_type(self, type_string, **kwargs)`



##### `serialize(self)`

Returns a serialized representation of current ScaleType

Returns
-------

### DeriveJunction



### Key



#### Methods

##### `ask(self, *args, **kwargs)`



##### `build(self, *args, **kwargs)`



##### `clone(self, repo: str, path: str = None, **kwargs)`



Type annotations:
```python
repo: <class 'str'>
path: <class 'str'>
```

##### `copy_module(self, module: str, path: str)`



Type annotations:
```python
module: <class 'str'>
path: <class 'str'>
```

##### `decrypt(self, data, password=None)`



##### `decrypt_message(self, encrypted_message_with_nonce: bytes, sender_public_key: bytes) -> bytes`

Decrypts message from a specified sender

Parameters
----------
encrypted_message_with_nonce: message to be decrypted
sender_public_key: sender's public key

Returns
-------
Decrypted message

Type annotations:
```python
encrypted_message_with_nonce: <class 'bytes'>
sender_public_key: <class 'bytes'>
return: <class 'bytes'>
```

##### `encrypt(self, data, password=None)`



##### `encrypt_message(self, message: Union[bytes, str], recipient_public_key: bytes, nonce: bytes = b'\x8f\x00x{z\x18?\x02*\xc3Q\x02\x8e\x9d\x17~\xc8\x19\\\xa0E\xe7\xdf\xf8') -> bytes`

Encrypts message with for specified recipient

Parameters
----------
message: message to be encrypted, bytes or string
recipient_public_key: recipient's public key
nonce: the nonce to use in the encryption

Returns
-------
Encrypted message

Type annotations:
```python
message: typing.Union[bytes, str]
recipient_public_key: <class 'bytes'>
nonce: <class 'bytes'>
return: <class 'bytes'>
```

##### `encrypted_keys(self)`



##### `ensure_sys_path()`



##### `epoch(self, *args, **kwargs)`



##### `export_to_encrypted_json(self, passphrase: str, name: str = None) -> dict`

Export Key to PolkadotJS format encrypted JSON file

Parameters
----------
passphrase: Used to encrypt the keypair
name: Display name of Key used

Returns
-------
dict

Type annotations:
```python
passphrase: <class 'str'>
name: <class 'str'>
return: <class 'dict'>
```

##### `file2hash(self, path='./')`



##### `fn_n(self, search=None)`



##### `forward(self, *args, **kwargs)`



##### `get_age(self, k: str) -> int`



Type annotations:
```python
k: <class 'str'>
return: <class 'int'>
```

##### `get_yaml(path: str = None, default={}, **kwargs) -> Dict`

fLoads a yaml file

Type annotations:
```python
path: <class 'str'>
return: typing.Dict
```

##### `has_module(self, path: str)`



Type annotations:
```python
path: <class 'str'>
```

##### `install(self, path)`



##### `is_error(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `is_repo(self, repo: str)`



Type annotations:
```python
repo: <class 'str'>
```

##### `is_ticket(self, data)`



##### `key2encrypted(self)`



##### `n_fns(self, search=None)`



##### `net(self)`



##### `print(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `progress(*args, **kwargs)`



##### `pull(self)`



##### `push(self, msg: str = 'update')`



Type annotations:
```python
msg: <class 'str'>
```

##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_encryption_data(self, data)`



##### `resolve_encryption_password(self, password: str = None) -> str`



Type annotations:
```python
password: <class 'str'>
return: <class 'str'>
```

##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `round(x, sig=6, small_value=1e-09)`



##### `save(self, path=None)`



##### `set_config(self, config: Union[str, dict, NoneType] = None) -> 'Munch'`

Set the config as well as its local params

Type annotations:
```python
config: typing.Union[str, dict, NoneType]
return: Munch
```

##### `set_crypto_type(self, crypto_type)`



##### `set_key(self, key: str, **kwargs) -> None`



Type annotations:
```python
key: <class 'str'>
return: None
```

##### `set_private_key(self, private_key: Union[bytes, str] = None, ss58_format: int = 42, crypto_type: int = 'sr25519', derive_path: str = None, path: str = None, **kwargs)`

Allows generation of Keys from a variety of input combination, such as a public/private key combination,
mnemonic or URI containing soft and hard derivation paths. With these Keys data can be signed and verified

Parameters
----------
ss58_address: Substrate address
public_key: hex string or bytes of public_key key
private_key: hex string or bytes of private key
ss58_format: Substrate address format, default to 42 when omitted
seed_hex: hex string of seed
crypto_type: Use KeyType.SR25519 or KeyType.ED25519 cryptography for generating the Key

Type annotations:
```python
private_key: typing.Union[bytes, str]
ss58_format: <class 'int'>
crypto_type: <class 'int'>
derive_path: <class 'str'>
path: <class 'str'>
```

##### `setattr(self, k, v)`



##### `sign(self, data: Union[scalecodec.base.ScaleBytes, bytes, str], to_json=False) -> bytes`

Creates a signature for given data
Parameters
----------
data: data to sign in `Scalebytes`, bytes or hex string format
Returns
-------
signature in bytes

Type annotations:
```python
data: typing.Union[scalecodec.base.ScaleBytes, bytes, str]
return: <class 'bytes'>
```

##### `sleep(period)`



##### `ss58_decode(*args, **kwargs)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `ss58_encode(*args, **kwargs)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `storage_migration(self)`



##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `to_json(self, password: str = None) -> dict`



Type annotations:
```python
password: <class 'str'>
return: <class 'dict'>
```

##### `tqdm(*args, **kwargs)`



##### `verify(self, data: Union[scalecodec.base.ScaleBytes, bytes, str, dict], signature: Union[bytes, str] = None, public_key: Optional[str] = None, return_address=False, ss58_format=42, max_age=None, address=None, **kwargs) -> bool`

Verifies data with specified signature

Parameters
----------
data: data to be verified in `Scalebytes`, bytes or hex string format
signature: signature in bytes or hex string format
public_key: public key in bytes or hex string format

Returns
-------
True if data is signed with this Key, otherwise False

Type annotations:
```python
data: typing.Union[scalecodec.base.ScaleBytes, bytes, str, dict]
signature: typing.Union[bytes, str]
public_key: typing.Optional[str]
return: <class 'bool'>
```

##### `vs(self, path=None)`



### KeyType



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



### ScaleBytes

Representation of SCALE encoded Bytes.

#### Methods

##### `get_next_bytes(self, length: int) -> bytearray`

Retrieve `length` amount of bytes of the stream

Parameters
----------
length: amount of requested bytes

Returns
-------
bytearray

Type annotations:
```python
length: <class 'int'>
return: <class 'bytearray'>
```

##### `get_remaining_bytes(self) -> bytearray`

Retrieves all remaining bytes from the stream

Returns
-------
bytearray

Type annotations:
```python
return: <class 'bytearray'>
```

##### `get_remaining_length(self) -> int`

Returns how many bytes are left in the stream

Returns
-------
int

Type annotations:
```python
return: <class 'int'>
```

##### `reset(self)`

Resets the pointer of the stream to the beginning

Returns
-------

##### `to_hex(self) -> str`

Return a hex-string (e.g. "0x00") representation of the byte-stream

Returns
-------
str

Type annotations:
```python
return: <class 'str'>
```

### SecretBox

The SecretBox class encrypts and decrypts messages using the given secret
key.

The ciphertexts generated by :class:`~nacl.secret.Secretbox` include a 16
byte authenticator which is checked as part of the decryption. An invalid
authenticator will cause the decrypt function to raise an exception. The
authenticator is not a signature. Once you've decrypted the message you've
demonstrated the ability to create arbitrary valid message, so messages you
send are repudiable. For non-repudiable messages, sign them after
encryption.

Encryption is done using `XSalsa20-Poly1305`_, and there are no practical
limits on the number or size of messages (up to 2⁶⁴ messages, each up to 2⁶⁴
bytes).

.. _XSalsa20-Poly1305: https://doc.libsodium.org/secret-key_cryptography/secretbox#algorithm-details

:param key: The secret key used to encrypt and decrypt messages
:param encoder: The encoder class used to decode the given key

:cvar KEY_SIZE: The size that the key is required to be.
:cvar NONCE_SIZE: The size that the nonce is required to be.
:cvar MACBYTES: The size of the authentication MAC tag in bytes.
:cvar MESSAGEBYTES_MAX: The maximum size of a message which can be
                        safely encrypted with a single key/nonce
                        pair.

#### Methods

##### `decrypt(self, ciphertext: bytes, nonce: Optional[bytes] = None, encoder: Type[nacl.encoding._Encoder] = <class 'nacl.encoding.RawEncoder'>) -> bytes`

Decrypts the ciphertext using the `nonce` (explicitly, when passed as a
parameter or implicitly, when omitted, as part of the ciphertext) and
returns the plaintext message.

:param ciphertext: [:class:`bytes`] The encrypted message to decrypt
:param nonce: [:class:`bytes`] The nonce used when encrypting the
    ciphertext
:param encoder: The encoder used to decode the ciphertext.
:rtype: [:class:`bytes`]

Type annotations:
```python
ciphertext: <class 'bytes'>
nonce: typing.Optional[bytes]
encoder: typing.Type[nacl.encoding._Encoder]
return: <class 'bytes'>
```

##### `encode(self: <class 'SupportsBytes'>, encoder: Type[nacl.encoding._Encoder] = <class 'nacl.encoding.RawEncoder'>) -> bytes`



Type annotations:
```python
self: <class 'typing.SupportsBytes'>
encoder: typing.Type[nacl.encoding._Encoder]
return: <class 'bytes'>
```

##### `encrypt(self, plaintext: bytes, nonce: Optional[bytes] = None, encoder: Type[nacl.encoding._Encoder] = <class 'nacl.encoding.RawEncoder'>) -> nacl.utils.EncryptedMessage`

Encrypts the plaintext message using the given `nonce` (or generates
one randomly if omitted) and returns the ciphertext encoded with the
encoder.

.. warning:: It is **VITALLY** important that the nonce is a nonce,
    i.e. it is a number used only once for any given key. If you fail
    to do this, you compromise the privacy of the messages encrypted.
    Give your nonces a different prefix, or have one side use an odd
    counter and one an even counter. Just make sure they are different.

:param plaintext: [:class:`bytes`] The plaintext message to encrypt
:param nonce: [:class:`bytes`] The nonce to use in the encryption
:param encoder: The encoder to use to encode the ciphertext
:rtype: [:class:`nacl.utils.EncryptedMessage`]

Type annotations:
```python
plaintext: <class 'bytes'>
nonce: typing.Optional[bytes]
encoder: typing.Type[nacl.encoding._Encoder]
return: <class 'nacl.utils.EncryptedMessage'>
```

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

### blake2b

Return a new BLAKE2b hash object.

## Functions

### `b64encode(s, altchars=None)`

Encode the bytes-like object s using Base64 and return a bytes object.

Optional altchars should be a byte string of length 2 which specifies an
alternative alphabet for the '+' and '/' characters.  This allows an
application to e.g. generate url or filesystem safe Base64 strings.

### `bip39seed_to_bip32masternode(seed)`



### `decode_pair_from_encrypted_json(json_data: Union[str, dict], passphrase: str) -> tuple`

Decodes encrypted PKCS#8 message from PolkadotJS JSON format

Parameters
----------
json_data
passphrase

Returns
-------
tuple containing private and public key

Type annotations:
```python
json_data: typing.Union[str, dict]
passphrase: <class 'str'>
return: <class 'tuple'>
```

### `decode_pkcs8(ciphertext: bytes) -> tuple`



Type annotations:
```python
ciphertext: <class 'bytes'>
return: <class 'tuple'>
```

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

### `encode_pair(public_key: bytes, private_key: bytes, passphrase: str) -> bytes`

Encode a public/private pair to PKCS#8 format, encrypted with provided passphrase

Parameters
----------
public_key: 32 bytes public key
private_key: 64 bytes private key
passphrase: passphrase to encrypt the PKCS#8 message

Returns
-------
(Encrypted) PKCS#8 message bytes

Type annotations:
```python
public_key: <class 'bytes'>
private_key: <class 'bytes'>
passphrase: <class 'str'>
return: <class 'bytes'>
```

### `encode_pkcs8(public_key: bytes, private_key: bytes) -> bytes`



Type annotations:
```python
public_key: <class 'bytes'>
private_key: <class 'bytes'>
return: <class 'bytes'>
```

### `eth_utils_keccak(primitive: Union[bytes, int, bool, NoneType] = None, hexstr: Optional[str] = None, text: Optional[str] = None) -> bytes`



Type annotations:
```python
primitive: typing.Union[bytes, int, bool, NoneType]
hexstr: typing.Optional[str]
text: typing.Optional[str]
return: <class 'bytes'>
```

### `extract_derive_path(derive_path: str)`



Type annotations:
```python
derive_path: <class 'str'>
```

### `get_ss58_format(ss58_address: str) -> int`

Returns the SS58 format for given SS58 address

Parameters
----------
ss58_address

Returns
-------
int

Type annotations:
```python
ss58_address: <class 'str'>
return: <class 'int'>
```

### `is_valid_ss58_address(value: str, valid_ss58_format: Optional[int] = None) -> bool`

Checks if given value is a valid SS58 formatted address, optionally check if address is valid for specified
ss58_format

Parameters
----------
value: value to checked
valid_ss58_format: if valid_ss58_format is provided the address must be valid for specified ss58_format (network) as well

Returns
-------
bool

Type annotations:
```python
value: <class 'str'>
valid_ss58_format: typing.Optional[int]
return: <class 'bool'>
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



### `scrypt(password: bytes, salt: bytes = b'', n: int = 1048576, r: int = 8, p: int = 1, maxmem: int = 33554432, dklen: int = 64) -> bytes`

Derive a cryptographic key using the scrypt KDF.

:raises nacl.exceptions.UnavailableError: If called when using a
    minimal build of libsodium.

Implements the same signature as the ``hashlib.scrypt`` implemented
in cpython version 3.6

Type annotations:
```python
password: <class 'bytes'>
salt: <class 'bytes'>
n: <class 'int'>
r: <class 'int'>
p: <class 'int'>
maxmem: <class 'int'>
dklen: <class 'int'>
return: <class 'bytes'>
```

### `ss58_decode(address: str, valid_ss58_format: Optional[int] = None) -> str`

Decodes given SS58 encoded address to an account ID
Parameters
----------
address: e.g. EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk
valid_ss58_format

Returns
-------
Decoded string AccountId

Type annotations:
```python
address: <class 'str'>
valid_ss58_format: typing.Optional[int]
return: <class 'str'>
```

### `ss58_encode(address: Union[str, bytes], ss58_format: int = 42) -> str`

Encodes an account ID to an Substrate address according to provided address_type

Parameters
----------
address
ss58_format

Returns
-------
str

Type annotations:
```python
address: typing.Union[str, bytes]
ss58_format: <class 'int'>
return: <class 'str'>
```

### `to_checksum_address(value: Union[~AnyAddress, str, bytes]) -> eth_typing.evm.ChecksumAddress`

Makes a checksum address given a supported format.

Type annotations:
```python
value: typing.Union[~AnyAddress, str, bytes]
return: eth_typing.evm.ChecksumAddress
```

