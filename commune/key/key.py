
import json
from typing import Union, Optional
import time
import os
import binascii
import re
import secrets
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import nacl.bindings
import nacl.public
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
from .utils import (extract_derive_path, 
                    python2str, 
                    ss58_encode, 
                    ss58_decode, get_ss58_format, 
                    is_valid_ss58_address,
                     b64encode, 
                     mnemonic_to_ecdsa_private_key, 
                     ecdsa_sign, 
                     ecdsa_verify)


class KeyType:
    ED25519 = 0
    SR25519 = 1
    ECDSA = 2


class Key(c.Module):
    default_key = 'module'
    Types = KeyType
    key_type_map = {k.lower():v for k,v in KeyType.__dict__.items()}
    ss58_format = 42
    key_type =  'sr25519'
    language_code = 'en'
    def __init__(self,
                 private_key: Union[bytes, str] = None, 
                 key_type: int = key_type,
                 path:str = None,
                 **kwargs): 
        self.set_private_key(private_key=private_key, 
                             key_type=key_type, **kwargs)

    @property
    def short_address(self):
        n = 4
        return self.ss58_address[:n] + '...' + self.ss58_address[-n:]
        
    def set_key_type(self, key_type):
        key_type = self.resolve_key_type(key_type)
        if key_type != self.key_type:
            kwargs = {
                'private_key': self.private_key,
                'ss58_format': self.ss58_format,
                'path': self.path, 
                'key_type': key_type # update key_type
            }
            return self.set_private_key(**kwargs)
        else:
            return {'success': False, 'message': f'key_type already set to {key_type}'}

    def set_private_key(self, 
                 private_key: Union[bytes, str] = None, 
                 key_type: int = key_type,
                 **kwargs
                 ):
        """
        Allows generation of Keys from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these Keys data can be signed and verified

        Parameters
        ----------
        ss58_address: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        seed_hex: hex string of seed
        key_type: Use KeyType.SR25519 or KeyType.ED25519 cryptography for generating the Key
        """
        key_type = self.resolve_key_type(key_type)
        # If no arguments are provided, generate a random keypair
        if  private_key == None:
            private_key = self.new_key(key_type=key_type).private_key
        if type(private_key) == str:
            private_key = c.str2bytes(private_key)
        key_type = self.resolve_key_type(key_type)
        if key_type == KeyType.SR25519:
            if len(private_key) != 64:
                private_key = sr25519.pair_from_seed(private_key)[1]
            public_key = sr25519.public_from_secret_key(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        elif key_type == KeyType.ED25519:       
            private_key = private_key[:32]    
            assert len(private_key) == 32  
            public_key, private_key = ed25519_zebra.ed_from_seed(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        elif key_type == KeyType.ECDSA:
            private_key = private_key[0:32]
            assert len(private_key) == 32
            private_key_obj = PrivateKey(private_key)
            public_key = private_key_obj.public_key.to_address()
            key_address = private_key_obj.public_key.to_checksum_address()
        else:
            raise ValueError('key_type "{}" not supported'.format(key_type))
        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))
        self.public_key = public_key
        self.address = self.key_address =  self.ss58_address = key_address
        self.private_key = private_key
        self.key_type = key_type
        self.key_type_name = self.key_type2name(self.key_type)
        return {'key_address':key_address, 'key_type':key_type}

    @classmethod
    def add_key(cls, path:str, mnemonic:str = None, password:str=None, refresh:bool=False, private_key=None, **kwargs):
        if cls.key_exists(path) and not refresh :
            c.print(f'key already exists at {path}')
            key_json = cls.get(path)
            if key_json != None:
                return cls.from_json(cls.get(path))
        key = cls.new_key(mnemonic=mnemonic, private_key=private_key, **kwargs)
        key_json = key.to_json()
        if password != None:
            key_json = cls.encrypt(data=key_json, password=password)
        c.print(cls.put(path, key_json))
        cls.update()
        return  json.loads(key_json)
    
    @classmethod
    def mv_key(cls, path, new_path):
        assert cls.key_exists(path), f'key does not exist at {path}'
        cls.put(new_path, cls.get_key(path).to_json())
        cls.rm_key(path)
        assert cls.key_exists(new_path), f'key does not exist at {new_path}'
        assert not cls.key_exists(path), f'key still exists at {path}'
        new_key = cls.get_key(new_path)
        return {'success': True, 'from': path , 'to': new_path, 'key': new_key}
    
    @classmethod
    def copy_key(cls, path, new_path):
        assert cls.key_exists(path), f'key does not exist at {path}'
        cls.put(new_path, cls.get_key(path).to_json())
        assert cls.key_exists(new_path), f'key does not exist at {new_path}'
        assert cls.get_key(path) == cls.get_key(new_path), f'key does not match'
        new_key = cls.get_key(new_path)
        return {'success': True, 'from': path , 'to': new_path, 'key': new_key}

    @classmethod
    def add_keys(cls, name, n=100, verbose:bool = False, **kwargs):
        response = []
        for i in range(n):
            key_name = f'{name}.{i}'
            if bool == True:
                c.print(f'generating key {key_name}')
            response.append(cls.add_key(key_name, **kwargs))

        return response
    
    def key2encrypted(self):
        keys = self.keys()
        key2encrypted = {}
        for k in keys:
            key2encrypted[k] = self.is_key_encrypted(k)
        return key2encrypted
    
    def encrypted_keys(self):
        return [k for k,v in self.key2encrypted().items() if v == True]
            
    @classmethod
    def key_info(cls, path='module', **kwargs):
        return cls.get_key_json(path)
    
    @classmethod
    def save_keys(cls, path='saved_keys.json', **kwargs):
        path = cls.resolve_path(path)
        c.print(f'saving mems to {path}')
        key2mnemonic = cls.key2mnemonic()
        c.put_json(path, key2mnemonic)
        return {'success': True, 'msg': 'saved keys', 'path':path, 'n': len(key2mnemonic)}
    
    @classmethod
    def load_keys(cls, path='saved_keys.json', refresh=False, **kwargs):
        key2mnemonic = c.get_json(path)
        for k,mnemonic in key2mnemonic.items():
            try:
                cls.add_key(k, mnemonic=mnemonic, refresh=refresh, **kwargs)
            except Exception as e:
                # c.print(f'failed to load mem {k} due to {e}', color='red')
                pass
        return {'loaded_mems':list(key2mnemonic.keys()), 'path':path}
    loadkeys = loadmems = load_keys
    
    @classmethod
    def key2mnemonic(cls, search=None) -> dict[str, str]:
        """
        keyname (str) --> mnemonic (str)
        
        """
        mems = {}
        for key in cls.keys(search):
            try:
                mems[key] = cls.get_mnemonic(key)
            except Exception as e:
                c.print(f'failed to get mem for {key} due to {e}')
        if search:
            mems = {k:v for k,v in mems.items() if search in k or search in v}
        return mems
    
    @classmethod
    def get_key(cls, 
                path:str,
                password:str=None, 
                create_if_not_exists:bool = True, 
                key_type=key_type, 
                **kwargs):
        for k in ['key_type', 'type']:
            if k in kwargs:
                key_type = kwargs.pop(k)
                break
        if hasattr(path, 'key_address'):
            key =  path
            return key
        path = path or 'module'
        # if ss58_address is provided, get key from address
        if cls.valid_ss58_address(path):
            path = cls.address2key().get(path)
        if not cls.key_exists(path):
            if create_if_not_exists:
                key = cls.add_key(path, **kwargs)
                c.print(f'key does not exist, generating new key -> {path}')
            else:
                raise ValueError(f'key does not exist at --> {path}')
        key_json = cls.get(path)
        # if key is encrypted, decrypt it
        if cls.is_encrypted(key_json):
            key_json = c.decrypt(data=key_json, password=password)
            if key_json == None:
                c.print({'status': 'error', 'message': f'key is encrypted, please {path} provide password'})
            return None
        key_json = json.loads(key_json) if isinstance(key_json, str) else key_json
        key =  cls.from_json(key_json, key_type=key_type)
        key.path = path
        return key
        
    @classmethod
    def get_keys(cls, search=None, clean_failed_keys=False):
        keys = {}
        for key in cls.keys():
            if str(search) in key or search == None:
                try:
                    keys[key] = cls.get_key(key)
                except Exception as e:
                    continue
                if keys[key] == None:
                    if clean_failed_keys:
                        cls.rm_key(key)
                    keys.pop(key) 
        return keys
        
    @classmethod
    def key2address(cls, search=None, max_age=10, update=False, **kwargs):
        path = 'key2address'
        key2address = cls.get(path, None, max_age=max_age, update=update)
        if key2address == None:
            key2address = {}
            for k,v in cls.get_keys(search).items():
                try:
                    key2address[k] = v.ss58_address
                except Exception as e:
                    c.print(f'failed to get address for {k} due to {e}')
            cls.put(path, key2address)
        return key2address
    
    @classmethod
    def n(cls, search=None, **kwargs):
        return len(cls.key2address(search, **kwargs))

    @classmethod
    def address2key(cls, search:Optional[str]=None, update:bool=False):
        address2key =  { v: k for k,v in cls.key2address(update=update).items()}
        if search != None :
            return address2key.get(search, None)
        return address2key
    
    @classmethod
    def get_address(cls, key):
        return cls.get_key(key).ss58_address
    get_addy = get_address
    @classmethod
    def key_paths(cls):
        return cls.ls()
    address_seperator = '_address='
    @classmethod
    def key2path(cls) -> dict:
        """
        defines the path for each key
        """
        path2key_fn = lambda path: '.'.join(path.split('/')[-1].split('.')[:-1])
        key2path = {path2key_fn(path):path for path in cls.key_paths()}
        return key2path

    @classmethod
    def keys(cls, search : str = None, **kwargs):
        keys = list(cls.key2path().keys())
        if search != None:
            keys = [key for key in keys if search in key]
        return keys

    @classmethod
    def n(cls, *args, **kwargs):
        return len(cls.key2address(*args, **kwargs))
    
    @classmethod
    def key_exists(cls, key, **kwargs):
        path = cls.get_key_path(key)
        return os.path.exists(path)
    
    @classmethod
    def get_key_path(cls, key):
        if not isinstance(key, str):
            key = Key.default_key
        key_path = cls.storage_dir() + '/' + key + '.json'
        return key_path
    
    @classmethod
    def get_key_json(cls, key):
        if key == None:
            key = Key.default_key
        print(f'getting key {key}')
        key_path =  cls.storage_dir() + '/' + key + '.json'
        return c.get(key_path)

    @classmethod
    def rm_key(cls, key=None):
        key2path = cls.key2path()
        keys = list(key2path.keys())
        if key not in keys:
            raise Exception(f'key {key} not found, available keys: {keys}')
        c.rm(key2path[key])
        return {'deleted':[key]}
        
    @classmethod
    def crypto_name2type(cls, name:str):
        key_type_map = cls.key_type_map 
        name = name.lower()
        if not name in key_type_map:
            raise ValueError(f'key_type {name} not supported {key_type_map}')
        return key_type_map[name]
        
    @classmethod
    def new_private_key(cls, key_type='ecdsa'):
        return cls.new_key(key_type=key_type).private_key.hex()
    
    @classmethod
    def new_key(cls, mnemonic:str = None, suri:str = None, private_key: str = None,
            key_type: Union[int,str] = 'sr25519', 
            **kwargs):
        '''
        yo rody, this is a class method you can gen keys whenever fam
        '''
        key_type = cls.resolve_key_type(key_type)
        if suri:
            key =  cls.from_uri(suri, key_type=key_type)
        elif mnemonic:
            key = cls.from_mnemonic(mnemonic, key_type=key_type)
        elif private_key:
            key = cls.from_private_key(private_key,key_type=key_type)
        else:
            key = cls.from_mnemonic(cls.generate_mnemonic(), key_type=key_type)
        return key
    
    create = gen = new_key
    
    def to_json(self, password: str = None ) -> dict:
        state_dict =  c.copy(self.__dict__)
        for k,v in state_dict.items():
            if type(v)  in [bytes]:
                state_dict[k] = v.hex() 
                if password != None:
                    state_dict[k] = self.encrypt(data=state_dict[k], password=password)
        if '_ss58_address' in state_dict:
            state_dict['ss58_address'] = state_dict.pop('_ss58_address')
        state_dict = json.dumps(state_dict)
        return state_dict
    
    @classmethod
    def from_json(cls, obj: Union[str, dict], password: str = None, key_type=None) -> dict:
        if type(obj) == str:
            obj = json.loads(obj)
        if obj == None:
           return None 
        if cls.is_encrypted(obj) and password != None:
            obj = cls.decrypt(data=obj, password=password)
        if 'ss58_address' in obj:
            obj['_ss58_address'] = obj.pop('ss58_address')
        if key_type != None:
            obj['key_type'] = key_type
        return  cls(**obj)

    @classmethod
    def generate_mnemonic(cls, words: int = 12) -> str:
        """
        params:
            words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24
        """
        mnemonic =  bip39_generate(words, cls.language_code)
        cls.validate_mnemonic(mnemonic)
        return mnemonic

    @classmethod
    def validate_mnemonic(cls, mnemonic: str) -> bool:
        """
        Verify if specified mnemonic is valid
        """
        assert bip39_validate(mnemonic, cls.language_code), """Invalid mnemonic, please provide a valid mnemonic"""


    @classmethod
    def from_mnemonic(cls, mnemonic: str = None, key_type=KeyType.SR25519) -> 'Key':
        """
        Create a Key for given memonic
        """
        if not mnemonic:
            mnemonic = cls.generate_mnemonic()
        if key_type == KeyType.ECDSA:
            if cls.language_code != "en":
                raise ValueError("ECDSA mnemonic only supports english")
            keypair = cls.from_private_key(mnemonic_to_ecdsa_private_key(mnemonic), key_type=key_type)
        else:
            keypair = cls.from_seed(
                seed_hex=binascii.hexlify(bytearray(bip39_to_mini_secret(mnemonic, "", cls.language_code))).decode("ascii"),
                key_type=key_type,
            )
        
        keypair.mnemonic = mnemonic

        return keypair

    from_mnemonic = from_mem = from_mnemonic

    @classmethod
    def from_seed(cls, seed_hex: Union[bytes, str], key_type=KeyType.SR25519) -> 'Key':
        """
        Create a Key for given seed

        Parameters
        ----------
        seed_hex: hex string of seed
        key_type: Use KeyType.SR25519 or KeyType.ED25519 cryptography for generating the Key

        Returns
        -------
        Key
        """
        key_type = cls.resolve_key_type(key_type)
        if type(seed_hex) is str:
            seed_hex = bytes.fromhex(seed_hex.replace('0x', ''))
        if key_type == KeyType.SR25519:
            public_key, private_key = sr25519.pair_from_seed(seed_hex)
        elif key_type == KeyType.ED25519:
            private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
        else:
            raise ValueError('key_type "{}" not supported'.format(key_type))
        
        ss58_address = ss58_encode(public_key, cls.ss58_format)

        kwargs =  dict(
            ss58_address=ss58_address, 
            public_key=public_key, 
            private_key=private_key,
            ss58_format=cls.ss58_format,
              key_type=key_type, 
        )
        
        return cls(**kwargs)
    @classmethod
    def from_password(cls, password:str, key_type=2, **kwargs):
        key= cls.from_uri(password, key_type=1, **kwargs)
        key.set_key_type(key_type)
        return key
    
    str2key = pwd2key = password2key = from_password = from_password

    @classmethod
    def from_uri(
            cls, 
            suri: str, 
            key_type=KeyType.SR25519, 
            DEV_PHRASE = 'bottom drive obey lake curtain smoke basket hold race lonely fit walk'

    ) -> 'Key':
        """
        Creates Key for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`

        Parameters
        ----------
        suri:
        key_type: Use KeyType.SR25519 or KeyType.ED25519 cryptography for generating the Key

        Returns
        -------
        Key
        """
        key_type = cls.resolve_key_type(key_type)
        suri = str(suri)
        if not suri.startswith('//'):
            suri = '//' + suri

        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri

        suri_regex = re.match(r'^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri)

        suri_parts = suri_regex.groupdict()

        if key_type == KeyType.ECDSA:
            private_key = mnemonic_to_ecdsa_private_key(
                mnemonic=suri_parts['phrase'],
                str_derivation_path=suri_parts['path'],
                passphrase=suri_parts['password']
            )
            derived_keypair = cls.from_private_key(private_key, key_type=key_type)
        else:

            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for key_type '{key_type}'")

            derived_keypair = cls.from_mnemonic(suri_parts['phrase'], key_type=key_type)

            if suri_parts['path'] != '':

                derived_keypair.derive_path = suri_parts['path']
                
                if key_type not in [KeyType.SR25519]:
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

                derived_keypair = Key(public_key=child_pubkey, private_key=child_privkey)

        return derived_keypair
    from_mnem = from_mnemonic = from_mnemonic
    @classmethod
    def from_private_key(
            cls, 
            private_key: Union[bytes, str],
            key_type: int = KeyType.SR25519
    ) -> 'Key':
        """
        Creates Key for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        key_type: Use KeyType.[SR25519|ED25519|ECDSA] cryptography for generating the Key
        Returns
        -------
        Key
        """
        return cls(private_key=private_key, key_type=key_type)

    def sign(self, data: Union[ScaleBytes, bytes, str], to_json = False, to_str=False) -> bytes:
        """
        Creates a signature for given data
        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format
        Returns
        -------
        signature in bytes

        """
        if not isinstance(data, str):
            data = python2str(data)
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()

        if not self.private_key:
            raise Exception('No private key set to create signatures')
        if self.key_type == KeyType.SR25519:
            signature = sr25519.sign((self.public_key, self.private_key), data)
        elif self.key_type == KeyType.ED25519:
            signature = ed25519_zebra.ed_sign(self.private_key, data)
        elif self.key_type == KeyType.ECDSA:
            signature = ecdsa_sign(self.private_key, data)
        else:
            raise Exception("Crypto type not supported")
        
        if to_json:
            return {'data':data.decode(),'key_type':self.key_type,'signature':signature.hex(),'address': self.ss58_address}
        if to_str:
            return '0x'+signature.hex()
        return signature

    def verify(self, 
               data: Union[ScaleBytes, bytes, str, dict], 
               signature: Union[bytes, str] = None,
               public_key:Optional[str]= None, 
               max_age = None,
               address = None,
               **kwargs
               ) -> bool:
        """
        Verifies data with specified signature

        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format
        public_key: public key in bytes or hex string format

        Returns
        -------
        True if data is signed with this Key, otherwise False
        """
        data = c.copy(data)

        if isinstance(data, dict):
            if 'data' in data and 'signature' in data and 'address' in data:
                signature = data.pop('signature')
                address = data.pop('address', address)
                data = data.pop('data')
            else:
                assert signature != None, 'signature not found in data'
                assert address != None, 'address not found in data'
       
        if max_age != None:
            if isinstance(data, int):
                staleness = c.timestamp() - int(data)
            elif 'timestamp' in data or 'time' in data:
                timestamp = data.get('timestamp', data.get('time'))
                staleness = c.timestamp() - int(timestamp)
            else:
                raise ValueError('data should be a timestamp or a dict with a timestamp key')
            assert staleness < max_age, f'data is too old, {staleness} seconds old, max_age is {max_age}'
        
        if not isinstance(data, str):
            data = python2str(data)
        if address != None:
            if self.valid_ss58_address(address):
                public_key = ss58_decode(address)
        if public_key == None:
            public_key = self.public_key
        if isinstance(public_key, str):
            public_key = bytes.fromhex(public_key.replace('0x', ''))
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()
        if type(signature) is str and signature[0:2] == '0x':
            signature = bytes.fromhex(signature[2:])
        elif type(signature) is str:
            signature = bytes.fromhex(signature)
        if type(signature) is not bytes:
            raise TypeError("Signature should be of type bytes or a hex-string")
        
        if self.key_type == KeyType.SR25519:
            crypto_verify_fn = sr25519.verify
        elif self.key_type == KeyType.ED25519:
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif self.key_type == KeyType.ECDSA:
            crypto_verify_fn = ecdsa_verify
        else:
            raise Exception("Crypto type not supported")
        verified = crypto_verify_fn(signature, data, public_key)
        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', public_key)
        return verified

    def resolve_encryption_password(self, password:str=None) -> str:
        if password == None:
            password = self.private_key
        if isinstance(password, str):
            password = password.encode()
        return hashlib.sha256(password).digest()

    def encrypt(self, data, password=None):
        if not isinstance(data, str):
            data = str(data)
        password = self.resolve_encryption_password(password)
        data = data + (AES.block_size - len(data) % AES.block_size) * chr(AES.block_size - len(data) % AES.block_size)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(password, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        return encrypted_bytes.decode() 

    def decrypt(self, data, password=None):    
        password = self.resolve_encryption_password(password)
        data = base64.b64decode(data)
        iv = data[:AES.block_size]
        cipher = AES.new(password, AES.MODE_CBC, iv)
        data =  cipher.decrypt(data[AES.block_size:])
        data = data[:-ord(data[len(data)-1:])].decode('utf-8')
        return data

    @classmethod
    def encrypt_key(cls, path = 'test.enc', password=None):
        assert cls.key_exists(path), f'file {path} does not exist'
        assert not cls.is_key_encrypted(path), f'{path} already encrypted'
        data = cls.get(path)
        enc_text = {'data': c.encrypt(data, password=password), 
                    'encrypted': True}
        cls.put(path, enc_text)
        return {'number_of_characters_encrypted':len(enc_text), 'path':path }
    
    @classmethod
    def is_key_encrypted(cls, key, data=None):
        data = data or cls.get(key)
        return cls.is_encrypted(data)
    
    @classmethod
    def decrypt_key(cls, path = 'test.enc', password=None, key=None):
        assert cls.key_exists(path), f'file {path} does not exist'
        assert cls.is_key_encrypted(path), f'{path} not encrypted'
        data = cls.get(path)
        assert cls.is_encrypted(data), f'{path} not encrypted'
        dec_text =  c.decrypt(data['data'], password=password)
        cls.put(path, dec_text)
        assert not cls.is_key_encrypted(path), f'failed to decrypt {path}'
        loaded_key = c.get_key(path)
        return { 'path':path , 'key_address': loaded_key.ss58_address,'key_type': loaded_key.key_type}

    @classmethod
    def get_mnemonic(cls, key):
        return cls.get_key(key).mnemonic

    def __str__(self):
        return f'<Key(address={self.key_address} type={self.key_type_name} path={self.path})>'
    
    def save(self, path=None):
        if path == None:
            path = self.path
        c.put_json(path, self.to_json())
        return {'saved':path}
        
    @classmethod
    def valid_ss58_address(cls, address: str ) -> bool:
        """
        Checks if the given address is a valid ss58 address.
        """
        try:
            return is_valid_ss58_address( address , valid_ss58_format=cls.ss58_format )
        except Exception as e:
            return False
          
    @classmethod
    def is_encrypted(cls, data):
        if isinstance(data, str):
            if os.path.exists(data):
                data = c.get_json(data)
            else:
                try:
                    data = json.loads(data)
                except: 
                    return False
        if isinstance(data, dict):
            return bool(data.get('encrypted', False))
        else:
            return False
        
    @classmethod
    def get_key_address(cls, key):
        return cls.get_key(key).ss58_address

    @classmethod
    def resolve_key_address(cls, key):
        key2address = c.key2address()
        if key in key2address:
            address = key2address[key]
        else:
            address = key
        return address

    def storage_migration(self): 
        key2path = self.key2path()
        new_key2path = {}
        for k_name, k_path in key2path.items():
            try:
                key = c.get_key(k_name)
                new_k_path = '/'.join(k_path.split('/')[:-1]) + '/' + f'{k_name}_address={key.ss58_address}_type={key.key_type}.json'
                new_key2path[k_name] = new_k_path
            except Exception as e:
                c.print(f'failed to migrate {k_name} due to {e}', color='red')
                
        return new_key2path

    @classmethod
    def crypto_name2type(cls, name:str):
        key_type_map = cls.key_type_map
        name = name.lower()
        if not name in key_type_map:
            raise ValueError(f'key_type {name} not supported {key_type_map}')
        return key_type_map[name]

    @classmethod
    def key_type2name(cls, key_type:str):
        key_type_map ={v:k for k,v  in cls.key_type_map.items()}
        return key_type_map[key_type]
    
    @classmethod
    def resolve_key_type(cls, key_type):
        if isinstance(key_type, int) or (isinstance(key_type, str) and c.is_int(key_type)):
            key_type = int(key_type)
            key_type_map = cls.key_type_map
            reverse_key_type_map = {v:k for k,v in key_type_map.items()}
            assert key_type in reverse_key_type_map, f'key_type {key_type} not supported {key_type_map}'
            key_type = reverse_key_type_map[key_type]
        if isinstance(key_type, str):
            key_type = key_type.lower()
            key_type = cls.crypto_name2type(key_type)
        return int(key_type)  



# if __name__ == "__main__":      
#     Key.run()







