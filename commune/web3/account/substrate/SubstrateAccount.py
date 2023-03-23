from substrateinterface import SubstrateInterface, Keypair
from typing import List, Dict, Union
import commune


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

import streamlit as st
__all__ = ['SubstrateAccount', 'SubstrateAccountType', 'MnemonicLanguageCode']


class SubstrateAccountType:
    """
    Type of cryptography, used in `SubstrateAccount` instance to encrypt and sign data
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

        
class SubstrateAccount(commune.Module):
    

    def __init__(self, 
                 ss58_address: str = None, 
                 seed_hex: Union[str, bytes] = None,
                 uri: str = None,
                 public_key: Union[bytes, str] = None,
                 private_key: Union[bytes, str] = None, 
                 ss58_format: int = 42, 
                 crypto_type: int = SubstrateAccountType.SR25519,
                 password: str = None,
                 mnemonic: str = None):
        """
        Allows generation of SubstrateAccounts from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these SubstrateAccounts data can be signed and verified
        Parameters
        ----------
        ss58_address: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        ss58_format: Substrate address format, default to 42 when omitted
        seed_hex: hex string of seed
        crypto_type: Use SubstrateAccountType.SR25519 or SubstrateAccountType.ED25519 cryptography for generating the SubstrateAccount
        """
        self.set_params(**self.get_params(locals()))
        
    def get_params(self, params):
        params.pop('self', None)
        return params
    
    def is_hex(self, x):
        if isinstance(x, bytes):
            try:
                x =x.decode('utf-8')
            except:
                x = x.hex()
        
        
        try:
            int(x, 16)
            return True
        except ValueError:
            return False
        except TypeError:
            return False
    
    def set_params(self, ss58_address: str = None, 
                 public_key: Union[bytes, str] = None,
                 private_key: Union[bytes, str] = None, 
                 ss58_format: int = None, 
                 seed_hex: Union[str, bytes] = None,
                 crypto_type: int = SubstrateAccountType.SR25519,
                 uri: str = None,
                 password: str = None,
                 mnemonic : str= None):
        
        
        if ss58_address == None and public_key == None and private_key == None and seed_hex == None and mnemonic == None:
            mnemonic = self.generate_mnemonic()

        params = self.get_params(locals())

        if mnemonic:
            mnemonic_data = self.create_from_mnemonic(mnemonic, data_only=True)
            params.update(mnemonic_data)
            
        elif seed_hex:
            seed_hex_data = self.create_from_seed(seed_hex = seed_hex, 
                                              ss58_format = ss58_format,
                                              crypto_type = crypto_type,
                                              data_only=True)
            params.update(seed_hex_data)

            
        # check if variable is a hex string
        
        
        self.crypto_type = crypto_type = params['crypto_type']
        self.seed_hex = params['seed_hex']
        private_key = params['private_key']
        public_key = params['public_key']
        ss58_address = params['ss58_address']
        ss58_format = params['ss58_format']
        password = params['password']
        self.mnemonic = mnemonic = params['mnemonic']
        seed_hex = params['seed_hex']
        


    
        self.params = params
        if crypto_type != SubstrateAccountType.ECDSA and ss58_address and not public_key:
            public_key = ss58_decode(ss58_address, 
                                     valid_ss58_format=ss58_format)

        if private_key:

            if type(private_key) is str:
                private_key = bytes.fromhex(private_key.replace('0x', ''))

            if self.crypto_type == SubstrateAccountType.SR25519:
                if len(private_key) != 64:
                    raise ValueError('Secret key should be 64 bytes long')
                if not public_key:
                   public_key = sr25519.public_from_secret_key(private_key)

            if self.crypto_type == SubstrateAccountType.ECDSA:
                private_key_obj = PrivateKey(private_key)
                public_key = private_key_obj.public_key.to_address()
                ss58_address = private_key_obj.public_key.to_checksum_address()


        if not public_key:
            raise ValueError('No SS58 formatted address or public key provided')

        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))

        if crypto_type == SubstrateAccountType.ECDSA:
            if len(public_key) != 20:
                raise ValueError('Public key should be 20 bytes long')
        else:
            if len(public_key) != 32:
                raise ValueError('Public key should be 32 bytes long')

            if not ss58_address:
                ss58_address = ss58_encode(public_key, ss58_format=ss58_format)

        self.__dict__.update(params)
        
            
        for key, value in params.items():
 
            if hasattr(value, 'hex'):
                params[key] = value.hex()
            elif isinstance(value, bytes):
                params[key] = value.decode()
        
        
        self.params = params
        
        self.set_password(password)


    set_keypair= set_params



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
    def create_from_mnemonic(cls, mnemonic: str,
                             ss58_format=42, 
                             crypto_type=SubstrateAccountType.SR25519,
                             language_code: str = MnemonicLanguageCode.ENGLISH,
                             data_only: bool = False) -> 'SubstrateAccount':
        """
        Create a SubstrateAccount for given memonic
        Parameters
        ----------
        mnemonic: Seed phrase
        ss58_format: Substrate address format
        crypto_type: Use `SubstrateAccountType.SR25519` or `SubstrateAccountType.ED25519` cryptography for generating the SubstrateAccount
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        SubstrateAccount
        """

        if crypto_type == SubstrateAccountType.ECDSA:
            if language_code != MnemonicLanguageCode.ENGLISH:
                raise ValueError("ECDSA mnemonic only supports english")

            private_key = mnemonic_to_ecdsa_private_key(mnemonic)
            keypair = cls.create_from_private_key(private_key, ss58_format=ss58_format, crypto_type=crypto_type)

        else:
            seed_array = bip39_to_mini_secret(mnemonic, "", language_code)

            keypair = cls.create_from_seed(
                seed_hex=binascii.hexlify(bytearray(seed_array)).decode("ascii"),
                ss58_format=ss58_format,
                crypto_type=crypto_type,
                data_only=data_only
            )
            
            if data_only:
                keypair['mnemonic'] = mnemonic
                return keypair
            
            
            

        keypair.mnemonic = mnemonic

        return keypair
    @classmethod
    def create_from_password(cls, password):
        seed = cls.hash(password)
        return cls.create_from_seed(seed)

    @classmethod
    def create_from_seed(
            cls, seed_hex: Union[bytes, str],
            ss58_format: Optional[int] = 42,
            crypto_type=SubstrateAccountType.SR25519,
            data_only: bool = False
    ) -> 'SubstrateAccount':
        """
        Create a SubstrateAccount for given seed
        Parameters
        ----------
        seed_hex: hex string of seed
        ss58_format: Substrate address format
        crypto_type: Use SubstrateAccountType.SR25519 or SubstrateAccountType.ED25519 cryptography for generating the SubstrateAccount
        Returns
        -------
        SubstrateAccount
        """

        if type(seed_hex) is str:
            # to hex bytes
            seed_hex = cls.hash(seed_hex, return_string=False)
         

        if crypto_type == SubstrateAccountType.SR25519:
            public_key, private_key = sr25519.pair_from_seed(seed_hex)
        elif crypto_type == SubstrateAccountType.ED25519:
            private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))

        ss58_address = ss58_encode(public_key, ss58_format)

        if data_only:
            return dict(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type, seed_hex=seed_hex
        )
        return cls(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type, seed_hex=seed_hex
        )

    @classmethod
    def create_from_uri(
            cls, suri: str, 
            ss58_format: Optional[int] = 42,
            crypto_type=SubstrateAccountType.SR25519,
            language_code: str = MnemonicLanguageCode.ENGLISH,
            return_data: bool = False
    ) -> 'SubstrateAccount':
        """
        Creates SubstrateAccount for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`
        Parameters
        ----------
        suri:
        ss58_format: Substrate address format
        crypto_type: Use SubstrateAccountType.SR25519 or SubstrateAccountType.ED25519 cryptography for generating the SubstrateAccount
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `MnemonicLanguageCode.ENGLISH`
        Returns
        -------
        SubstrateAccount
        """

        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri

        suri_regex = re.match(r'^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri)

        suri_parts = suri_regex.groupdict()

        if crypto_type == SubstrateAccountType.ECDSA:
            if language_code != MnemonicLanguageCode.ENGLISH:
                raise ValueError("ECDSA mnemonic only supports english")

            private_key = mnemonic_to_ecdsa_private_key(
                mnemonic=suri_parts['phrase'],
                str_derivation_path=suri_parts['path'][1:],
                passphrase=suri_parts['password'] or ''
            )
            derived_keypair = cls.create_from_private_key(private_key, ss58_format=ss58_format, crypto_type=crypto_type)
        else:

            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for crypto_type '{crypto_type}'")

            derived_keypair = cls.create_from_mnemonic(
                suri_parts['phrase'], ss58_format=ss58_format, crypto_type=crypto_type, language_code=language_code
            )

            if suri_parts['path'] != '':

                derived_keypair.derive_path = suri_parts['path']

                if crypto_type not in [SubstrateAccountType.SR25519]:
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
                        
                key_kwargs = dict(public_key=child_pubkey, private_key=child_privkey, ss58_format=ss58_format)
                if return_data:
                    return key_kwargs
                    

                derived_keypair = cls(key_kwargs)

        return derived_keypair

    @classmethod
    def create_from_private_key(
            cls, private_key: Union[bytes, str], public_key: Union[bytes, str] = None, ss58_address: str = None,
            ss58_format: int = None, crypto_type: int = SubstrateAccountType.SR25519
    ) -> 'SubstrateAccount':
        """
        Creates SubstrateAccount for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        public_key: hex string or bytes of public key
        ss58_address: Substrate address
        ss58_format: Substrate address format, default = 42
        crypto_type: Use SubstrateAccountType.[SR25519|ED25519|ECDSA] cryptography for generating the SubstrateAccount
        Returns
        -------
        SubstrateAccount
        """

        return cls(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type
        )

    @classmethod
    def create_from_encrypted_json(cls, json_data: Union[str, dict], passphrase: str,
                                   ss58_format: int = None) -> 'SubstrateAccount':
        """
        Create a SubstrateAccount from a PolkadotJS format encrypted JSON file
        Parameters
        ----------
        json_data: Dict or JSON string containing PolkadotJS export format
        passphrase: Used to encrypt the keypair
        ss58_format: Which network ID to use to format the SS58 address (42 for testnet)
        Returns
        -------
        SubstrateAccount
        """

        if type(json_data) is str:
            json_data = json.loads(json_data)

        private_key, public_key = decode_pair_from_encrypted_json(json_data, passphrase)

        if 'sr25519' in json_data['encoding']['content']:
            crypto_type = SubstrateAccountType.SR25519
        elif 'ed25519' in json_data['encoding']['content']:
            crypto_type = SubstrateAccountType.ED25519
            # Strip the nonce part of the private key
            private_key = private_key[0:32]
        else:
            raise NotImplementedError("Unknown SubstrateAccountType found in JSON")

        if ss58_format is None and 'address' in json_data:
            ss58_format = get_ss58_format(json_data['address'])

        return cls.create_from_private_key(private_key, public_key, ss58_format=ss58_format, crypto_type=crypto_type)

    def export_to_encrypted_json(self, passphrase: str, name: str = None) -> dict:
        """
        Export SubstrateAccount to PolkadotJS format encrypted JSON file
        Parameters
        ----------
        passphrase: Used to encrypt the keypair
        name: Display name of SubstrateAccount used
        Returns
        -------
        dict
        """
        if not name:
            name = self.ss58_address

        if self.crypto_type != SubstrateAccountType.SR25519:
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

    def sign(self, data: Union[ScaleBytes, bytes, str],
             return_dict:bool = False, 
             return_string: bool = True) -> bytes:
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

        if self.crypto_type == SubstrateAccountType.SR25519:
            signature = sr25519.sign((self.public_key, self.private_key), data)

        elif self.crypto_type == SubstrateAccountType.ED25519:
            signature = ed25519_zebra.ed_sign(self.private_key, data)

        elif self.crypto_type == SubstrateAccountType.ECDSA:
            signature = ecdsa_sign(self.private_key, data)

        else:
            raise ConfigurationError("Crypto type not supported")

            
        if return_string:
            signature = self.python2str(signature)
        if return_dict:
            return {
                'data': data.decode(),
                'signature': signature,
                'public_key': self.public_key.hex(),
            }

        
        return signature

    def verify(self, data: Union[ScaleBytes, bytes, str], signature: Union[bytes, str] = None, public_key: str = None) -> bool:
        """
        Verifies data with specified signature
        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format
        Returns
        -------
        True if data is signed with this SubstrateAccount, otherwise False
        """
        if isinstance(data, dict) and 'signature' in data:
            signature = data['signature']
            public_key = data['public_key']
            data = data['data']
            
        public_key =  public_key if public_key else self.public_key

        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))
        if not isinstance(data, str):
            data = self.python2str(data)

        if type(data) is ScaleBytes:
            data = bytes(data.data)
            
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()

        if type(signature) is str:
            signature = bytes.fromhex(signature.replace('0x', ''))

        if type(signature) is not bytes:
            raise TypeError("Signature should be of type bytes or a hex-string")

        if self.crypto_type == SubstrateAccountType.SR25519:
            crypto_verify_fn = sr25519.verify
        elif self.crypto_type == SubstrateAccountType.ED25519:
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif self.crypto_type == SubstrateAccountType.ECDSA:
            crypto_verify_fn = ecdsa_verify
        else:
            raise ConfigurationError("Crypto type not supported")

        verified = crypto_verify_fn(signature, data, public_key)

        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', public_key)

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
        if self.crypto_type != SubstrateAccountType.ED25519:
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
        if self.crypto_type != SubstrateAccountType.ED25519:
            raise ConfigurationError('Only ed25519 keypair type supported')
        private_key = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(self.private_key + self.public_key)
        recipient = nacl.public.PrivateKey(private_key)
        curve25519_public_key = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(sender_public_key)
        sender = nacl.public.PublicKey(curve25519_public_key)
        return nacl.public.Box(recipient, sender).decrypt(encrypted_message_with_nonce)

    def __repr__(self):
        if self.ss58_address:
            return '<SubstrateAccount (address={})>'.format(self.ss58_address)
        else:
            return '<SubstrateAccount (public_key=0x{})>'.format(self.public_key.hex())
        


    @property
    def address(self):
        return self.ss58_address

    @classmethod
    def from_uri(cls, uri):
        """ Create a SubstrateAccount from a URI.
        """
        if not uri.startswith('/'):
            uri = '/' + uri
        
        keypair =  cls(keypair=cls.create_from_uri(uri))
        # keypair = cls.create_from_uri(uri)

        return keypair

    @classmethod
    def test_accounts(cls, demo_uris:List[str] = ['alice', 'bob', 'chris', 'billy', 'dave', 'sarah']) -> Dict[str, 'SubstrateAccount']:
        '''
        This method is used to create demo accounts for testing purposes.
        '''
        
        demo_accounts = {}
        for demo_uri in demo_uris:
            demo_accounts[demo_uri] =  cls.create_from_password(demo_uri)
            
        
        return demo_accounts 


    def set_password(self, password: str = None) -> 'AESKey':
        if password == None:
            if  self.private_key != None:
                if type(self.private_key) is str:
                    self.private_key = bytes.fromhex(self.private_key.replace('0x', ''))
                password = self.private_key.hex()
            elif self.public_key != None:
                password = self.public_key.hex()
            else:
                raise ValueError("No password or private/public key provided")
            
        self.password = password
        aes_seed = self.hash(self.password)
        
        # get the AES key module and create an instance for encryption
        aes_key = commune.get_module('crypto.key.aes')
        self.aes_key = aes_key(aes_seed)
        
    @classmethod
    def hash(cls, data: Union[str, bytes], **kwargs) -> bytes:
        if not hasattr(cls, 'hash_module'):
            cls.hash_module = commune.get_module('crypto.hash')()
        return cls.hash_module(data, **kwargs)
    
    
    
    def encrypt(self, data: Union[str, bytes], password:str = None) -> bytes:
        self.set_password(password)
        return self.aes_key.encrypt(data)
    
    def decrypt(self, data: Union[str, bytes], password: str = None) -> bytes:
        self.set_password(password)
        return self.aes_key.decrypt(data)
    
    
    def state_dict(self, password: str = None, encrypt: bool = True) -> dict:
        from copy import deepcopy
        state_dict = {'data': {}, 'encrypted': encrypt}   
        state_dict['data'] = deepcopy(self.params)
        for k in ['public_key', 'private_key', 'seed_hex']:
            if isinstance(self.params[k], bytes):
                state_dict['data'][k] = self.params[k].hex()
        
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
        import streamlit as st
        
        encrypted = state.get('encrypted', False)
        if encrypted == True:
            state = self.decrypt(data=state['data'], password=password)
        else:
            state = state['data']
        self.params = state
        self.set_params(**self.params)
        
    def save(self, path: str,  password: str = None, encrypt: bool = True):
        state = self.encrypt(data=state['data'], password=password, encrypt=encrypt)
        self.put_json(path, state)

    def load(self, path: str, password: str = None):
        state = self.get_json(path)
        self.load_state_dict(state=state, password=password)

    @classmethod
    def test_state_dict(cls):
        import streamlit as st
        password = 'hello'
        self = SubstrateAccount(password=password)
        
        self2 = SubstrateAccount(password=password)
        # testing recontraction of encrypted state
        self.state_dict()
        self2.load_state_dict(self.state_dict())
        self2.address == self.address
        
        self.state_dict(encrypt=False)
        self2.load_state_dict(self.state_dict())
        assert self2.address == self.address
      
    @classmethod  
    def test_save_loader(cls, password: str = None):
        # cls.test_state_dict()
        
        self  = cls()
        self.save(password=password)
        self.load(password=password)
        
    
    @classmethod
    def test(cls):
        for fn in dir(cls):
            if fn.startswith('test_'):
                getattr(cls, fn)()
        
    def save( self, path: str = 'default', encrypt:bool = True, password: str = None):
        
        if not password:
            password = self.password
            
        state_dict = self.state_dict(password=password, encrypt=encrypt)
        self.put_json(path, state_dict)
        
    def load( self, path: str = 'default', password: str = None):
        
        if not password:
            password = self.password
            
        state_dict = self.get_json(path)
        self.load_state_dict(state_dict, password=password)
        

        
    
if __name__ == '__main__':
    SubstrateAccount.test()

    

