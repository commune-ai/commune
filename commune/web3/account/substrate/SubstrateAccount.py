

# Python Substrate Interface Library
#
# Copyright 2018-2023 Stichting Polkascan (Polkascan Foundation).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from scalecodec.utils.ss58 import ss58_encode, ss58_decode, get_ss58_format

from scalecodec.base import ScaleBytes
from typing import Union, Optional

import time
import binascii
import re
import secrets
from base64 import b64encode

import nacl.bindings
import nacl.public
from eth_keys.datatypes import PrivateKey

from substrateinterface.constants import DEV_PHRASE
from substrateinterface.exceptions import ConfigurationError
from substrateinterface.key import extract_derive_path
from substrateinterface.utils.ecdsa_helpers import mnemonic_to_ecdsa_private_key, ecdsa_verify, ecdsa_sign
from substrateinterface.utils.encrypted_json import decode_pair_from_encrypted_json, encode_pair

from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
import ed25519_zebra

__all__ = ['Keypair', 'KeypairType', 'MnemonicLanguageCode']


class KeypairType:
    """
    Type of cryptography, used in `Keypair` instance to encrypt and sign data
    * ED25519 = 0
    * SR25519 = 1
    * ECDSA = 2
    """
    ED25519 = 0
    SR25519 = 1
    ECDSA = 2


class MnemonicLanguageCode:
    """
    Available language codes to generate mnemonics
    * ENGLISH = 'en'
    * CHINESE_SIMPLIFIED = 'zh-hans'
    * CHINESE_TRADITIONAL = 'zh-hant'
    * FRENCH = 'fr'
    * ITALIAN = 'it'
    * JAPANESE = 'ja'
    * KOREAN = 'ko'
    * SPANISH = 'es'
    """
    ENGLISH = 'en'
    CHINESE_SIMPLIFIED = 'zh-hans'
    CHINESE_TRADITIONAL = 'zh-hant'
    FRENCH = 'fr'
    ITALIAN = 'it'
    JAPANESE = 'ja'
    KOREAN = 'ko'
    SPANISH = 'es'
import commune

class Keypair(commune.Module):

    def __init__(self, 
                 key: str = None,
                 ss58_address: str = None,
                 public_key: Union[bytes, str] = None,
                 private_key: Union[bytes, str] = None,
                 ss58_format: int = None, 
                 seed_hex: Union[str, bytes] = None,
                 mnemonic: str = None,
                 uri: str = None,
                 derive_path : str = None,
                 crypto_type: int = KeypairType.SR25519):
        params = locals()
        params.pop('self')
        self.set_params(**params)
    def set_params(self, 
                 key: str = None,
                 ss58_address: str = None,
                 public_key: Union[bytes, str] = None,
                 private_key: Union[bytes, str] = None,
                 ss58_format: int = None, 
                 uri: str = None,
                 seed_hex: Union[str, bytes] = None,
                 mnemonic: str = None,
                 derive_path : str = None,
                
                 crypto_type: int = KeypairType.SR25519):
        """
        Allows generation of Keypairs from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these Keypairs data can be signed and verified
        Parameters
        ----------
        ss58_address: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        ss58_format: Substrate address format, default to 42 when omitted
        seed_hex: hex string of seed
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair
        """
        if key == None and ss58_address == None and public_key == None and private_key == None and seed_hex == None:
            key = 'bitconnect'
        if key:
            if isinstance(key, str):
                seed_hex = self.hash(key)
            
        if seed_hex != None: 

            kwargs = self.create_from_seed(seed_hex, return_dict=True)
            self.__dict__.update(kwargs)
        elif mnemonic != None:
            
            kwargs = self.create_from_mnemonic(mnemonic, return_dict=True)
            self.__dict__.update(kwargs)
            
        elif uri != None:
            kwargs = self.create_from_uri(uri, return_dict=True)
            self.__dict__.update(kwargs)
        else:
            self.ss58_address = ss58_address
            self.public_key = public_key
            self.private_key = private_key
            self.ss58_format = ss58_format
            self.seed_hex = seed_hex
            self.mnemonic = mnemonic
            self.derive_path  = derive_path
            self.crypto_type =  crypto_type
            
        self.derive_path = None


        if self.crypto_type != KeypairType.ECDSA and self.ss58_address and not self.public_key:
            self.public_key = ss58_decode(self.ss58_address, valid_ss58_format=self.ss58_format)

        if self.private_key:

            if type(self.private_key) is str:
                self.private_key = bytes.fromhex(self.private_key.replace('0x', ''))

            if self.crypto_type == KeypairType.SR25519:
                if len(self.private_key) != 64:
                    raise ValueError('Secret key should be 64 bytes long')
                if not public_key:
                    self.public_key = sr25519.public_from_secret_key(self.private_key)

            if self.crypto_type == KeypairType.ECDSA:
                private_key_obj = PrivateKey(self.private_key)
                self.public_key = private_key_obj.public_key.to_address()
                self.ss58_address = private_key_obj.public_key.to_checksum_address()

        if not self.public_key:
            raise ValueError('No SS58 formatted address or public key provided')

        if type(self.public_key) is str:
            self.public_key = bytes.fromhex(self.public_key.replace('0x', ''))

        if self.crypto_type == KeypairType.ECDSA:
            if len(self.public_key) != 20:
                raise ValueError('Public key should be 20 bytes long')
        else:
            if len(self.public_key) != 32:
                raise ValueError('Public key should be 32 bytes long')

            if not self.ss58_address:
                self.ss58_address = ss58_encode(self.public_key, ss58_format=self.ss58_format)



    @classmethod
    def generate_mnemonic(cls, words: int = 12, language_code: str = MnemonicLanguageCode.ENGLISH) -> str:
        """
        Generates a new seed phrase with given amount of words (default 12)
        Parameters
        ----------
        words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        str: Seed phrase
        """
        return bip39_generate(words, language_code)

    @classmethod
    def validate_mnemonic(cls, mnemonic: str, language_code: str = MnemonicLanguageCode.ENGLISH) -> bool:
        """
        Verify if specified mnemonic is valid
        Parameters
        ----------
        mnemonic: Seed phrase
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        bool
        """
        return bip39_validate(mnemonic, language_code)

    @classmethod
    def create_from_mnemonic(cls, mnemonic: str = None, 
                             ss58_format=42, 
                             crypto_type=KeypairType.SR25519,
                             language_code: str = MnemonicLanguageCode.ENGLISH,
                             return_dict : bool = True) -> 'Keypair':
        """
        Create a Keypair for given memonic
        Parameters
        ----------
        mnemonic: Seed phrase
        ss58_format: Substrate address format
        crypto_type: Use `KeypairType.SR25519` or `KeypairType.ED25519` cryptography for generating the Keypair
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        Keypair
        """
        if mnemonic == None:
            mnemonic = cls.generate_mnemonic(language_code=language_code)

        if crypto_type == KeypairType.ECDSA:
            if language_code != MnemonicLanguageCode.ENGLISH:
                raise ValueError("ECDSA mnemonic only supports english")

            private_key = mnemonic_to_ecdsa_private_key(mnemonic)
            keypair_kwargs = dict(private_key=private_key, ss58_format=ss58_format, crypto_type=crypto_type)

            keypair = cls.create_from_private_key(**kwargs)

        else:
            seed_array = bip39_to_mini_secret(mnemonic, "", language_code)
            keypair_kwargs = dict(
                seed_hex=binascii.hexlify(bytearray(seed_array)).decode("ascii"),
                ss58_format=ss58_format,
                crypto_type=crypto_type
            )
            
            
                
            keypair = cls.create_from_seed(**keypair_kwargs)

        keypair.mnemonic = mnemonic

        return keypair

    @classmethod
    def create_from_seed(
            cls, seed_hex: Union[bytes, str],
            ss58_format: Optional[int] = 42,
            crypto_type=KeypairType.SR25519,
            return_dict: bool = False
    ) -> 'Keypair':
        """
        Create a Keypair for given seed
        Parameters
        ----------
        seed_hex: hex string of seed
        ss58_format: Substrate address format
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair
        Returns
        -------
        Keypair
        """

        if type(seed_hex) is str:
            
            seed_hex = bytes.fromhex(seed_hex.replace('0x', ''))

        
        if crypto_type == KeypairType.SR25519:
            public_key, private_key = sr25519.pair_from_seed(seed_hex)

        elif crypto_type == KeypairType.ED25519:
            private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))

        ss58_address = ss58_encode(public_key, ss58_format)
        
        cls_kwargs = dict(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type, seed_hex=seed_hex
        )
        if return_dict:
            return cls_kwargs
            
        return cls(**cls_kwargs)

    @classmethod
    def create_from_uri(
            cls, 
            suri: str, 
            ss58_format: Optional[int] = 42, 
            crypto_type=KeypairType.SR25519, 
            language_code: str = MnemonicLanguageCode.ENGLISH,
            return_dict: bool = False
    ) -> 'Keypair':
        """
        Creates Keypair for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`
        Parameters
        ----------
        suri:
        ss58_format: Substrate address format
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        Keypair
        """

        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri

        suri_regex = re.match(r'^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri)

        suri_parts = suri_regex.groupdict()

        if crypto_type == KeypairType.ECDSA:
            if language_code != MnemonicLanguageCode.ENGLISH:
                raise ValueError("ECDSA mnemonic only supports english")

            private_key = mnemonic_to_ecdsa_private_key(
                mnemonic=suri_parts['phrase'],
                str_derivation_path=suri_parts['path'][1:],
                passphrase=suri_parts['password'] or ''
            )
            kwargs = dict(private_key=private_key, ss58_format=ss58_format, crypto_type=crypto_type)
            if return_dict:
                return kwargs
            
            derived_keypair = cls.create_from_private_key(kwargs)
        else:

            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for crypto_type '{crypto_type}'")

            derived_keypair = cls.create_from_mnemonic(
                suri_parts['phrase'],
                ss58_format=ss58_format, 
                crypto_type=crypto_type, 
                language_code=language_code,
                return_dict=return_dict
            )
            if return_dict:
                assert isinstance(derived_keypair, dict)
                kwargs = derived_keypair
                return kwargs

            if suri_parts['path'] != '':

                derived_keypair.derive_path = suri_parts['path']

                if crypto_type not in [KeypairType.SR25519]:
                    raise NotImplementedError('Derivation paths for this crypto type not supported')

                derive_junctions = extract_derive_path(suri_parts['path'])

                child_pubkey = derived_keypair.public_key
                child_privkey = derived_keypair.private_key

                for junction in derive_junctions:

                    if junction.is_hard:

                        _, child_pubkey, child_privkey = sr25519.hard_derive_keypair(
                            (junction.chain_code, child_pubkey, child_privkey),
                            b''
                        )

                    else:

                        _, child_pubkey, child_privkey = sr25519.derive_keypair(
                            (junction.chain_code, child_pubkey, child_privkey),
                            b''
                        )
                if return_dict:
                    return kwargs
                
                derived_keypair = Keypair(**kwargs)

        return derived_keypair

    @classmethod
    def create_from_private_key(
            cls, private_key: Union[bytes, str], public_key: Union[bytes, str] = None, ss58_address: str = None,
            ss58_format: int = None, crypto_type: int = KeypairType.SR25519
    ) -> 'Keypair':
        """
        Creates Keypair for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        public_key: hex string or bytes of public key
        ss58_address: Substrate address
        ss58_format: Substrate address format, default = 42
        crypto_type: Use KeypairType.[SR25519|ED25519|ECDSA] cryptography for generating the Keypair
        Returns
        -------
        Keypair
        """

        return cls(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type
        )

    @classmethod
    def create_from_encrypted_json(cls, json_data: Union[str, dict], passphrase: str,
                                   ss58_format: int = None) -> 'Keypair':
        """
        Create a Keypair from a PolkadotJS format encrypted JSON file
        Parameters
        ----------
        json_data: Dict or JSON string containing PolkadotJS export format
        passphrase: Used to encrypt the keypair
        ss58_format: Which network ID to use to format the SS58 address (42 for testnet)
        Returns
        -------
        Keypair
        """

        if type(json_data) is str:
            json_data = json.loads(json_data)

        private_key, public_key = decode_pair_from_encrypted_json(json_data, passphrase)

        if 'sr25519' in json_data['encoding']['content']:
            crypto_type = KeypairType.SR25519
        elif 'ed25519' in json_data['encoding']['content']:
            crypto_type = KeypairType.ED25519
            # Strip the nonce part of the private key
            private_key = private_key[0:32]
        else:
            raise NotImplementedError("Unknown KeypairType found in JSON")

        if ss58_format is None and 'address' in json_data:
            ss58_format = get_ss58_format(json_data['address'])

        return cls.create_from_private_key(private_key, public_key, ss58_format=ss58_format, crypto_type=crypto_type)

    def export_to_encrypted_json(self, passphrase: str, name: str = None) -> dict:
        """
        Export Keypair to PolkadotJS format encrypted JSON file
        Parameters
        ----------
        passphrase: Used to encrypt the keypair
        name: Display name of Keypair used
        Returns
        -------
        dict
        """
        if not name:
            name = self.ss58_address

        if self.crypto_type != KeypairType.SR25519:
            raise NotImplementedError(f"Cannot create JSON for crypto_type '{self.crypto_type}'")

        # Secret key from PolkadotJS is an Ed25519 expanded secret key, so has to be converted
        # https://github.com/polkadot-js/wasm/blob/master/packages/wasm-crypto/src/rs/sr25519.rs#L125
        converted_private_key = sr25519.convert_secret_key_to_ed25519(self.private_key)

        encoded = encode_pair(self.public_key, converted_private_key, passphrase)

        json_data = {
            "encoded": b64encode(encoded).decode(),
            "encoding": {"content": ["pkcs8", "sr25519"], "type": ["scrypt", "xsalsa20-poly1305"], "version": "3"},
            "address": self.ss58_address,
            "meta": {
                "name": name, "tags": [], "whenCreated": int(time.time())
            }
        }

        return json_data

    def sign(self, data: Union[ScaleBytes, bytes, str], return_dict=False) -> bytes:
        """
        Creates a signature for given data
        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format
        Returns
        -------
        signature in bytes
        """
        data = self.python2str(data)
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()

        if not self.private_key:
            raise ConfigurationError('No private key set to create signatures')

        if self.crypto_type == KeypairType.SR25519:
            signature = sr25519.sign((self.public_key, self.private_key), data)

        elif self.crypto_type == KeypairType.ED25519:
            signature = ed25519_zebra.ed_sign(self.private_key, data)

        elif self.crypto_type == KeypairType.ECDSA:
            signature = ecdsa_sign(self.private_key, data)

        else:
            raise ConfigurationError("Crypto type not supported")

        if return_dict:
            return {
                'data': data.decode(),
                'signature': signature.hex(),
                'public_key': self.public_key.hex(),
                'ss58_address': self.ss58_address,
            }
        return signature

    def verify(self, data: Union[ScaleBytes, bytes, str],
               signature: Union[bytes, str] = None,
               public_key: str = None,
               ss58_address: str = None,
               return_public_key : bool = True) -> Union[bool, str]:
        """
        Verifies data with specified signature
        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format
        Returns
        -------
        True if data is signed with this Keypair, otherwise False
        """
        
        
        if signature == None:
            assert type(data) is dict, "If no signature is provided, data should be a dict"
            assert 'data' in data, "If no signature is provided, data should be a dict with 'data' key"
            assert 'signature' in data
            assert 'public_key' in data
            public_key = data['public_key']
            signature = data['signature']
            data = data['data']
        if isinstance(signature, str):
            signature = bytes.fromhex(signature)
            
        if not isinstance(data, str):
            data = self.python2str(data)
        
        if public_key == None:
            public_key = self.public_key.hex()
            
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()

        if type(signature) is str and signature[0:2] == '0x':
            signature = bytes.fromhex(signature[2:])

        if type(signature) is not bytes:
            raise TypeError("Signature should be of type bytes or a hex-string")

        if self.crypto_type == KeypairType.SR25519:
            crypto_verify_fn = sr25519.verify
        elif self.crypto_type == KeypairType.ED25519:
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif self.crypto_type == KeypairType.ECDSA:
            crypto_verify_fn = ecdsa_verify
        else:
            raise ConfigurationError("Crypto type not supported")

        verified = crypto_verify_fn(signature, data, self.public_key)

        if return_public_key:
            public_key = self.public_key.hex()
        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', self.public_key)

        return verified

    def encrypt_message(
        self, message: Union[bytes, str], recipient_public_key: bytes, nonce: bytes = secrets.token_bytes(24),
    ) -> bytes:
        """
        Encrypts message with for specified recipient
        Parameters
        ----------
        message: message to be encrypted, bytes or string
        recipient_public_key: recipient's public key
        nonce: the nonce to use in the encryption
        Returns
        -------
        Encrypted message
        """

        if not self.private_key:
            raise ConfigurationError('No private key set to encrypt')
        if self.crypto_type != KeypairType.ED25519:
            raise ConfigurationError('Only ed25519 keypair type supported')
        curve25519_public_key = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(recipient_public_key)
        recipient = nacl.public.PublicKey(curve25519_public_key)
        private_key = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(self.private_key + self.public_key)
        sender = nacl.public.PrivateKey(private_key)
        box = nacl.public.Box(sender, recipient)
        return box.encrypt(message if isinstance(message, bytes) else message.encode("utf-8"), nonce)

    def decrypt_message(self, encrypted_message_with_nonce: bytes, sender_public_key: bytes) -> bytes:
        """
        Decrypts message from a specified sender
        Parameters
        ----------
        encrypted_message_with_nonce: message to be decrypted
        sender_public_key: sender's public key
        Returns
        -------
        Decrypted message
        """

        if not self.private_key:
            raise ConfigurationError('No private key set to decrypt')
        if self.crypto_type != KeypairType.ED25519:
            raise ConfigurationError('Only ed25519 keypair type supported')
        private_key = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(self.private_key + self.public_key)
        recipient = nacl.public.PrivateKey(private_key)
        curve25519_public_key = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(sender_public_key)
        sender = nacl.public.PublicKey(curve25519_public_key)
        return nacl.public.Box(recipient, sender).decrypt(encrypted_message_with_nonce)

    def __repr__(self):
        if self.ss58_address:
            return '<Keypair (address={})>'.format(self.ss58_address)
        else:
            return '<Keypair (public_key=0x{})>'.format(self.public_key.hex())


        
    def set_aes_key(self, password: str = None):

        self.password = password
        self.aes_seed = self.hash(self.password)
        
        # get the AES key module and create an instance for encryption
        aes_key = commune.get_module('crypto.key.aes')
        self.aes_key = aes_key(self.aes_seed)
        
    
    def encrypt(self, data: Union[str, bytes], password:str = None) -> bytes:
        aes_key = self.resolve_aes_key(password)
        return aes_key.encrypt(data)
    
            
        # return cls.create_from_uri(uri)
    
    def decrypt(self, data: Union[str, bytes], password: str = None) -> bytes:
        aes_key = self.resolve_aes_key(password)
        return aes_key.decrypt(data)
    
    
    def resolve_password(self, password: str = None) -> str:
        
        if password == None:
            if hasattr(self, 'password') and self.password != None:
                password = self.password
            
            elif  self.private_key != None:
                if type(self.private_key) is str:
                    self.private_key = bytes.fromhex(self.private_key.replace('0x', ''))
                password = self.private_key.hex()
            elif self.public_key != None:
                if type(self.public_key) is str:
                    self.public_key = bytes.fromhex(self.public_key.replace('0x', ''))
                password = self.public_key.hex()
            else:
                raise ValueError("No password or private/public key provided")
            
        assert isinstance(password, str), "Password must be a string"
        
        password = self.hash(password)
        return password
    def resolve_aes_key(self, password: str = None) -> 'commune.crypto.key.aes':
        
        password = self.resolve_password(password)
        key = commune.get_key(password, mode='aes')
        return key
            
    
    def state_dict(self, password: str = None, encrypt: bool = False ) -> dict:
        from copy import deepcopy
        state_dict = {'data': {}, 'encrypted': encrypt}   
        state_dict['data'] =self.__dict__ 

        for k,v in state_dict['data'].items():
            if type(v)  in [bytes]:
                state_dict['data'][k] = v.hex()
                
        print(state_dict['data'])
        if encrypt == True:
            state_dict['data'] = self.encrypt(data=state_dict['data'], password=password)
            
        
        return state_dict

    def load_state_dict(self, state: dict, password: str = None):
        
        '''
        
        We assume that the state dict is encrypted if the key 'encrypted' is set to True.
        We also assume that the data is encrypted as bytes
        
        Example of state dict:
            state = {'data': b'encrypted_data', 'encrypted': True}
  
        '''
        
        encrypted = state.get('encrypted', False)
        if encrypted == True:
            state = self.decrypt(data=state['data'], password=password)
        else:
            state = state['data']
        self.set_params(**state)
           
    def save(self, path: str,  password: str = None, encrypt: bool = True):
        state = self.encrypt(data=state['data'], password=password, encrypt=encrypt)
        self.put_json(path, state)

    def load(self, path: str, password: str = None):
        state = self.get_json(path)
        self.load_state_dict(state=state, password=password)
        
    @classmethod
    def load_from_dict(cls, state: dict, password: str = None):
        return cls(**state)
