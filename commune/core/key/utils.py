
import json
from typing import Union, Optional
import time
import os
import re
import base64
from base64 import b64encode
import hashlib
from scalecodec.utils.ss58 import ss58_encode, ss58_decode, get_ss58_format, is_valid_ss58_address
from scalecodec.base import ScaleBytes
from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
import ed25519_zebra
import commune as c
import re
from hashlib import blake2b
import base64
import json
from os import urandom
from typing import Union
from nacl.hashlib import scrypt
from nacl.secret import SecretBox
from sr25519 import pair_from_ed25519_secret_key
from scalecodec.types import Bytes
import hashlib
import hmac
import struct
from eth_keys.datatypes import Signature, PrivateKey
from eth_utils import to_checksum_address, keccak as eth_utils_keccak
from ecdsa.curves import SECP256k1

BIP39_PBKDF2_ROUNDS = 2048
BIP39_SALT_MODIFIER = "mnemonic"
BIP32_PRIVDEV = 0x80000000
BIP32_CURVE = SECP256k1
BIP32_SEED_MODIFIER = b'Bitcoin seed'
ETH_DERIVATION_PATH = "m/44'/60'/0'/0"
JUNCTION_ID_LEN = 32
RE_JUNCTION = r'(\/\/?)([^/]+)'


def is_valid_ss58_address(address: str) -> bool:
    """
    Check if the given address is a valid SS58 address
    """
    try:
        ss58_decode(address)
        return True
    except:
        return False



def str2bytes( data: str, mode: str = 'hex') -> bytes:
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        return bytes.fromhex(data)


def is_int(value: str) -> bool:
    """
    Check if the provided string is an integer
    """
    try:
        int(value)
        return True
    except ValueError:
        return False

def is_valid_ecdsa_address(address: str) -> bool:
    """
    Check if the given address is a valid ECDSA address
    """
    try:
        return len(bytes.fromhex(address)) == 20
    except:
        return False
class PublicKey:
    def __init__(self, private_key):
        self.point = int.from_bytes(private_key, byteorder='big') * BIP32_CURVE.generator

    def __bytes__(self):
        xstr = int(self.point.x()).to_bytes(32, byteorder='big')
        parity = int(self.point.y()) & 1
        return (2 + parity).to_bytes(1, byteorder='big') + xstr

    def address(self):
        x = int(self.point.x())
        y = int(self.point.y())
        s = x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        return to_checksum_address(eth_utils_keccak(s)[12:])

def mnemonic_to_bip39seed(mnemonic, passphrase):
    mnemonic = bytes(mnemonic, 'utf8')
    salt = bytes(BIP39_SALT_MODIFIER + passphrase, 'utf8')
    return hashlib.pbkdf2_hmac('sha512', mnemonic, salt, BIP39_PBKDF2_ROUNDS)

def bip39seed_to_bip32masternode(seed):
    h = hmac.new(BIP32_SEED_MODIFIER, seed, hashlib.sha512).digest()
    key, chain_code = h[:32], h[32:]
    return key, chain_code

def derive_bip32childkey(parent_key, parent_chain_code, i):
    assert len(parent_key) == 32
    assert len(parent_chain_code) == 32
    k = parent_chain_code
    if (i & BIP32_PRIVDEV) != 0:
        key = b'\x00' + parent_key
    else:
        key = bytes(PublicKey(parent_key))
    d = key + struct.pack('>L', i)
    while True:
        h = hmac.new(k, d, hashlib.sha512).digest()
        key, chain_code = h[:32], h[32:]
        a = int.from_bytes(key, byteorder='big')
        b = int.from_bytes(parent_key, byteorder='big')
        key = (a + b) % int(BIP32_CURVE.order)
        if a < BIP32_CURVE.order and key != 0:
            key = key.to_bytes(32, byteorder='big')
            break
        d = b'\x01' + h[32:] + struct.pack('>L', i)
    return key, chain_code

def parse_derivation_path(str_derivation_path):
    path = []
    if str_derivation_path[0:2] != 'm/':
        raise ValueError("Can't recognize derivation path. It should look like \"m/44'/60/0'/0\".")
    for i in str_derivation_path.lstrip('m/').split('/'):
        if "'" in i:
            path.append(BIP32_PRIVDEV + int(i[:-1]))
        else:
            path.append(int(i))
    return path


def mnemonic_to_ecdsa_private_key(mnemonic: str, str_derivation_path: str = None, passphrase: str = "") -> bytes:

    if str_derivation_path is None:
        str_derivation_path = f'{ETH_DERIVATION_PATH}/0'

    derivation_path = parse_derivation_path(str_derivation_path)
    bip39seed = mnemonic_to_bip39seed(mnemonic, passphrase)
    master_private_key, master_chain_code = bip39seed_to_bip32masternode(bip39seed)
    private_key, chain_code = master_private_key, master_chain_code
    for i in derivation_path:
        private_key, chain_code = derive_bip32childkey(private_key, chain_code, i)
    return private_key


def ecdsa_sign(private_key: bytes, message: bytes) -> bytes:
    signer = PrivateKey(private_key)
    return signer.sign_msg(message).to_bytes()


def ecdsa_verify(signature: bytes, data: bytes, address: bytes) -> bool:
    signature_obj = Signature(signature)
    recovered_pubkey = signature_obj.recover_public_key_from_msg(data)
    return recovered_pubkey.to_canonical_address() == address

NONCE_LENGTH = 24
SCRYPT_LENGTH = 32 + (3 * 4)
PKCS8_DIVIDER = bytes([161, 35, 3, 33, 0])
PKCS8_HEADER = bytes([48, 83, 2, 1, 1, 48, 5, 6, 3, 43, 101, 112, 4, 34, 4, 32])
PUB_LENGTH = 32
SALT_LENGTH = 32
SEC_LENGTH = 64
SEED_LENGTH = 32
SCRYPT_N = 1 << 15
SCRYPT_P = 1
SCRYPT_R = 8

def decode_pair_from_encrypted_json(json_data: Union[str, dict], passphrase: str) -> tuple:
    """
    Decodes encrypted PKCS#8 message from PolkadotJS JSON format

    Parameters
    ----------
    json_data
    passphrase

    Returns
    -------
    tuple containing private and public key
    """
    if type(json_data) is str:
        json_data = json.loads(json_data)

    # Check requirements
    if json_data.get('encoding', {}).get('version') != "3":
        raise ValueError("Unsupported JSON format")

    encrypted = base64.b64decode(json_data['encoded'])

    if 'scrypt' in json_data['encoding']['type']:
        salt = encrypted[0:32]
        n = int.from_bytes(encrypted[32:36], byteorder='little')
        p = int.from_bytes(encrypted[36:40], byteorder='little')
        r = int.from_bytes(encrypted[40:44], byteorder='little')

        password = scrypt(passphrase.encode(), salt, n=n, r=r, p=p, dklen=32, maxmem=2 ** 26)
        encrypted = encrypted[SCRYPT_LENGTH:]

    else:
        password = passphrase.encode().rjust(32, b'\x00')

    if "xsalsa20-poly1305" not in json_data['encoding']['type']:
        raise ValueError("Unsupported encoding type")

    nonce = encrypted[0:NONCE_LENGTH]
    message = encrypted[NONCE_LENGTH:]

    secret_box = SecretBox(key=password)
    decrypted = secret_box.decrypt(message, nonce)

    # Decode PKCS8 message
    secret_key, public_key = decode_pkcs8(decrypted)

    if 'sr25519' in json_data['encoding']['content']:
        # Secret key from PolkadotJS is an Ed25519 expanded secret key, so has to be converted
        # https://github.com/polkadot-js/wasm/blob/master/packages/wasm-crypto/src/rs/sr25519.rs#L125
        converted_public_key, secret_key = pair_from_ed25519_secret_key(secret_key)
        assert(public_key == converted_public_key)

    return secret_key, public_key

def decode_pkcs8(ciphertext: bytes) -> tuple:
    current_offset = 0

    header = ciphertext[current_offset:len(PKCS8_HEADER)]
    if header != PKCS8_HEADER:
        raise ValueError("Invalid Pkcs8 header found in body")

    current_offset += len(PKCS8_HEADER)

    secret_key = ciphertext[current_offset:current_offset + SEC_LENGTH]
    current_offset += SEC_LENGTH

    divider = ciphertext[current_offset:current_offset + len(PKCS8_DIVIDER)]

    if divider != PKCS8_DIVIDER:
        raise ValueError("Invalid Pkcs8 divider found in body")

    current_offset += len(PKCS8_DIVIDER)

    public_key = ciphertext[current_offset: current_offset + PUB_LENGTH]

    return secret_key, public_key

def encode_pkcs8(public_key: bytes, private_key: bytes) -> bytes:
    return PKCS8_HEADER + private_key + PKCS8_DIVIDER + public_key

def encode_pair(public_key: bytes, private_key: bytes, passphrase: str) -> bytes:
    """
    Encode a public/private pair to PKCS#8 format, encrypted with provided passphrase

    Parameters
    ----------
    public_key: 32 bytes public key
    private_key: 64 bytes private key
    passphrase: passphrase to encrypt the PKCS#8 message

    Returns
    -------
    (Encrypted) PKCS#8 message bytes
    """
    message = encode_pkcs8(public_key, private_key)

    salt = urandom(SALT_LENGTH)
    password = scrypt(passphrase.encode(), salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32, maxmem=2 ** 26)

    secret_box = SecretBox(key=password)
    message = secret_box.encrypt(message)

    scrypt_params = SCRYPT_N.to_bytes(4, 'little') + SCRYPT_P.to_bytes(4, 'little') + SCRYPT_R.to_bytes(4, 'little')

    return salt + scrypt_params + message.nonce + message.ciphertext




def python2str(x):
    from copy import deepcopy
    import json
    x = deepcopy(x)
    input_type = type(x)
    if input_type == str:
        return x
    if input_type in [dict]:
        x = json.dumps(x)
    elif input_type in [bytes]:
        x = bytes2str(x)
    elif input_type in [list, tuple, set]:
        x = json.dumps(list(x))
    elif input_type in [int, float, bool]:
        x = str(x)
        
    return x

class DeriveJunction:
    def __init__(self, chain_code, is_hard=False):
        self.chain_code = chain_code
        self.is_hard = is_hard

    @classmethod
    def from_derive_path(cls, path: str, is_hard=False):

        if path.isnumeric():
            byte_length = ceil(int(path).bit_length() / 8)
            chain_code = int(path).to_bytes(byte_length, 'little').ljust(32, b'\x00')

        else:
            path_scale = Bytes()
            path_scale.encode(path)

            if len(path_scale.data) > JUNCTION_ID_LEN:
                chain_code = blake2b(path_scale.data.data, digest_size=32).digest()
            else:
                chain_code = bytes(path_scale.data.data.ljust(32, b'\x00'))

        return cls(chain_code=chain_code, is_hard=is_hard)

def extract_derive_path(derive_path: str):

    path_check = ''
    junctions = []
    paths = re.findall(RE_JUNCTION, derive_path)

    if paths:
        path_check = ''.join(''.join(path) for path in paths)

        for path_separator, path_value in paths:
            junctions.append(DeriveJunction.from_derive_path(
                path=path_value, is_hard=path_separator == '//')
            )

    if path_check != derive_path:
        raise ValueError('Reconstructed path "{}" does not match input'.format(path_check))

    return junctions

def bytes2str(data: bytes, mode: str = 'utf-8') -> str:
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)



def valid_h160_address(cls, address):
    # Check if it starts with '0x'
    if not address.startswith('0x'):
        return False
    
    # Remove '0x' prefix
    address = address[2:]
    
    # Check length
    if len(address) != 40:
        return False
    
    # Check if it contains only valid hex characters
    if not re.match('^[0-9a-fA-F]{40}$', address):
        return False
    
    return True


def is_mnemonic(mnemonic:str) -> bool:
    """
    Check if the provided string is a valid mnemonic
    """
    if not isinstance(mnemonic, str):
        return False
    return bip39_validate(mnemonic, self.language_code)