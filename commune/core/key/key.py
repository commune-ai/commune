
from typing import Union, Optional
import time
import os
import binascii
import re
import secrets
import base64
import hashlib
import nacl.bindings
import copy
import nacl.public
from scalecodec.base import ScaleBytes
from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
from sr25519 import pair_from_ed25519_secret_key
import ed25519_zebra
import commune as c
import re
from hashlib import blake2b
import json
from scalecodec.types import Bytes
import hashlib
from copy import deepcopy
import hmac
import copy
from Crypto import Random
import hashlib
from Crypto.Cipher import AES
import copy
import base64
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
                     str2bytes,
                     ecdsa_verify, is_int)

# imoport 
class Key:
    crypto_type_map = {'ed25519': 0, 'sr25519': 1, 'ecdsa': 2}
    crypto_type2networks = {
        'sr25519': ['dot', 'comai', 'com', 'bt'],
        'ecdsa': ['eth', 'btc']
    }
    crypto_types = list(crypto_type_map.keys())
    reverse_crypto_type_map = {v:k for k,v in crypto_type_map.items()}
    default_key= 'module'
    ss58_format = 42
    crypto_type =  'sr25519'
    language_code = 'en'

    def __init__(self,
                 private_key: Union[bytes, str] = None, 
                 mnemonic : Optional[str] = None,
                 crypto_type: int = crypto_type,
                 path:str = None,
                 storage_path = '~/.commune/key',
                 **kwargs): 
        self.storage_path = c.abspath(storage_path)
        self.set_private_key(private_key=private_key, crypto_type=crypto_type, mnemonic=mnemonic, **kwargs)

    def set_private_key(self, private_key: Union[bytes, str] ,  crypto_type: int , mnemonic:Optional[str] = None, **kwargs):
        """
        Allows generation of Keys from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these Keys data can be signed and verified

        Parameters
        ----------
        private_key: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        seed_hex: hex string of seed
        crypto_type: Use "sr25519" or "ed25519"cryptography for generating the Key
        """
        crypto_type = self.get_crypto_type(crypto_type)
        if  mnemonic:
            private_key = self.from_mnemonic(mnemonic, crypto_type=crypto_type).private_key
        elif private_key is None:
            # generate a new keypair if no private key is provided
            private_key = self.new_key(crypto_type=crypto_type).private_key
        if type(private_key) == str:
            private_key = str2bytes(private_key)

        # If the private key is longer than 32 bytes, it is assumed to be a seed and the public key is derived from it
        if crypto_type == 'sr25519':
            if len(private_key) != 64:
                private_key = sr25519.pair_from_seed(private_key)[1]
            public_key = sr25519.public_from_secret_key(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        elif crypto_type == "ecdsa":
            private_key = private_key[0:32] if len(private_key) > 32 else private_key
            assert len(private_key) == 32, f'private_key should be 32 bytes, got {len(private_key)}'
            private_key_obj = PrivateKey(private_key)
            public_key = private_key_obj.public_key.to_address()
            key_address = private_key_obj.public_key.to_checksum_address()
        elif crypto_type == "ed25519":       
            private_key = private_key[:32]    
            assert len(private_key) == 32  
            public_key, private_key = ed25519_zebra.ed_from_seed(private_key)
            key_address = ss58_encode(public_key, ss58_format=self.ss58_format)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))
        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))
            
        self.crypto_type_name = crypto_type # the name of the crypto type
        self.crypto_type = self.crypto_type_id = self.crypto_type_map[crypto_type] # the integer value of the crypto type
        self.private_key = private_key
        self.public_key = public_key
        self.key_address = self.address = self.ss58_address =  key_address
        self.multiaddress = self.multi()
        return {'key_address':key_address, 'crypto_type':crypto_type}

    def get_crypto_type(self, crypto_type=None):
        if crypto_type == None:
            crypto_type = self.crypto_type
        shortcuts = {}
        for k, net_list in self.crypto_type2networks.items():
            for net in net_list:
                shortcuts[net] = k
        if crypto_type in shortcuts:
            crypto_type = shortcuts[crypto_type]
        
        if is_int(crypto_type):
            crypto_type = self.reverse_crypto_type_map[int(crypto_type)]
        elif isinstance(crypto_type, str):
            crypto_type = crypto_type.lower()
        else: 
            raise ValueError(f'crypto_type {crypto_type} not supported')
        return crypto_type

    @property
    def shorty(self):
        n = 4
        return self.key_address[:n] + '...' + self.key_address[-n:]
        
    def valid_ss58_address(self, address):
        return is_valid_ss58_address(address)

    def add_key(self, path:str,  crypto_type=None, mnemonic:str = None, refresh:bool=False, private_key=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        if not self.key_exists(path, crypto_type=crypto_type) or refresh :
            key = self.new_key( private_key=private_key, crypto_type=crypto_type, mnemonic=mnemonic, **kwargs)
            key_json = json.loads(key.to_json())
            assert crypto_type == self.get_crypto_type(key_json['crypto_type']), f'crypto_type mismatch {crypto_type} != {key_json["crypto_type"]}'
            path = self.resolve_path(path) + '/' + crypto_type+ '/' + key.key_address + '.json'
            c.put(path, key_json)
            assert self.key_exists(path, crypto_type=crypto_type), f'key does not exist at {path}'
        return self.get_key(path, crypto_type=crypto_type)
    
    def mv_key(self, path, new_path):
        new_path = self.get_key_path(new_path)
        key = self.get_key(path)
        key_json = key.to_json()
        new_key_path = new_path + '/' + self.get_crypto_type(key.crypto_type) + '/' + key.key_address + '.json'
        new_key_path_dir = '/'.join(new_key_path.split('/')[:-1])
        if not c.exists(new_key_path_dir):
            os.makedirs(new_key_path_dir)
        c.put(new_key_path, key_json)
        old_key_path = self.get_key_path(path)
        c.rm(old_key_path)
        assert self.key_exists(new_key_path), f'key does not exist at {new_key_path}'
        assert not self.key_exists(old_key_path), f'key still exists at {old_key_path}'
        return {'success': True, 'from': path , 'to': new_path}

    def key2encrypted(self):
        return {k: self.is_key_encrypted(k) for k in self.keys()}
            
    def resolve_path(self, path:str) -> str:
        path = str(path)
        if not path.startswith(self.storage_path):
            path = self.storage_path + '/' + path
        return path

    def root_key(self):
        return self.get_key(self.default_key)

    def get_key(self, 
                path:str,
                password:Optional[str]=None, 
                create_if_not_exists:bool = True, 
                prompt_password:bool = False,
                crypto_type=None, 
                **kwargs):
        
        crypto_type = self.get_crypto_type(crypto_type)

        if hasattr(path, 'key_address'):
            return path

        if 'type' in kwargs:
            crypto_type = kwargs.pop('type')
        path = path or 'module'
        if not self.key_exists(path):
            if create_if_not_exists:
                key = self.add_key(path, **kwargs) # create key
            else:
                raise ValueError(f'key does not exist at --> {path}')
        key_json = self.get_key_data(path)
        if self.is_encrypted(key_json):
            if prompt_password and password == None:
                password = input(f'enter password to decrypt {path} ')
            key_json = self.decrypt(data=key_json, password=password)
        key_json = json.loads(key_json) if isinstance(key_json, str) else key_json
        key =  self.from_json(key_json, crypto_type=crypto_type)
        return key

    def get_keys(self, search=None, clean_failed_keys=False):
        keys = {}
        for key in self.keys():
            if str(search) in key or search == None:
                try:
                    keys[key] = self.get_key(key)
                except Exception as e:
                    continue
                if keys[key] == None:
                    if clean_failed_keys:
                        self.rm_key(key)
                    keys.pop(key) 
        return keys

    def key2path(self, crypto_type=crypto_type) -> dict:
        """
        defines the path for each key
        """
        crypto_type = self.get_crypto_type(crypto_type)
        key_paths  = c.ls(self.storage_path)
        key2path = {}
        for p in key_paths:
            files = c.glob(f'{p}/{crypto_type}')
            if len(files) >= 1:
                file2age = {f:os.path.getmtime(f) for f in files}
                files = [k for k,v in sorted(file2age.items(), key=lambda item: item[1])]
                # get the latest file
                p = files[0]
                # delete the others
                for f in files[1:]:
                    os.remove(f)
                name = p.split('/')[-3]
                key2path[name] = p         
        return key2path
    
    def key2address(self, search=None, crypto_type=None,  **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = self.key2path(crypto_type=crypto_type)
        key2address = {}
        for key, path in key2path.items():
            key2address[key] = path.split('/')[-1].split('.')[0]
        return key2address

    def key2type(self, search=None, crypto_type=None,  **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = self.key2path(crypto_type=crypto_type)
        key2address = {}
        for key, path in key2path.items():
            key2address[key] = path.split('/')[-1].split('.')[0]
        return key2address

    def address2key(self, search:Optional[str]=None,  crypto_type=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        address2key =  { v: k for k,v in self.key2address(crypto_type=crypto_type).items()}
        if search != None :
            return {k:v for k,v in address2key.items() if search in k}
        return address2key
    
    def keys(self, search : str = None, crypto_type=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        keys = list(self.key2path(crypto_type=crypto_type).keys())
        if search != None:
            keys = [key for key in keys if search in key]
        return keys
    
    def n(self, *args, **kwargs):
        return len(self.key2address(*args, **kwargs))
    
    def key_exists(self, key, crypto_type=None, **kwargs):
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = self.key2path(crypto_type=crypto_type)
        if f'/{crypto_type}/' in key:
            key = key.split(f'/')[-3]
        if key in key2path or key in key2path.values():
            return True
        return False
    
    def get_key_path(self, key, crypto_type=None):
        crypto_type = self.get_crypto_type(crypto_type)
        key2path = self.key2path(crypto_type=crypto_type)
        if key in key2path:
            return key2path[key]
        elif key in key2path.values():
            return key
        else:
            return self.resolve_path(key)

    def get_key_data(self, key, crypto_type=None):
        crypto_type = self.get_crypto_type(crypto_type)
        key_path =  self.get_key_path(key, crypto_type=crypto_type)
        output =  c.get(key_path)
        # if single quoted json, convert to double quoted json string and load
        if isinstance(output, str):
            output = output.replace("'", '"')
        return json.loads(output) if isinstance(output, str) else output

    def rm_key(self, key=None, crypto_type=None, **kwargs):
        key2path = self.key2path(crypto_type=crypto_type)
        keys = list(key2path.keys())
        if key not in keys:
            if key in key2path.values():
                key = [k for k,v in key2path.items() if v == key][0]
            else:
                raise Exception(f'key {key} not found, available keys: {keys}')
        c.rm(key2path[key])
        return {'deleted':[key]}

    def is_mnemonic(self, mnemonic:str) -> bool:
        """
        Check if the provided string is a valid mnemonic
        """
        if not isinstance(mnemonic, str):
            return False
        return bip39_validate(mnemonic, self.language_code)
    
    def new_key(self, mnemonic:str = None, suri:str = None, private_key: str = None, crypto_type: Union[int,str] = crypto_type,  **kwargs):
        '''
        yo rody, this is a class method you can gen keys whenever fam
        '''
        crypto_type = self.get_crypto_type(crypto_type)
        if mnemonic:
            key = self.from_mnemonic(mnemonic, crypto_type=crypto_type)
        elif private_key:
            key = self.from_private_key(private_key,crypto_type=crypto_type)
        elif suri:
            key =  self.from_uri(suri, crypto_type=crypto_type)
        else:
            key = self.from_mnemonic(self.generate_mnemonic(), crypto_type=crypto_type)
            
        return key
        
    def to_json(self, password: str = None ) -> dict:
        state_dict =  copy.deepcopy(self.__dict__)
        for k,v in state_dict.items():
            if type(v)  in [bytes]:
                state_dict[k] = v.hex() 
                if password != None:
                    state_dict[k] = self.encrypt(data=state_dict[k], password=password)
        if '_ss58_address' in state_dict:
            state_dict['ss58_address'] = state_dict.pop('_ss58_address')
        state_dict = json.dumps(state_dict)
        return state_dict
    
    def from_json(self, obj: Union[str, dict], password: str = None, crypto_type=None) -> dict:
        if type(obj) == str:
            obj = json.loads(obj)
        if obj == None:
           return None 
        if self.is_encrypted(obj) and password != None:
            obj = self.decrypt(data=obj, password=password)
        if 'ss58_address' in obj:
            obj['_ss58_address'] = obj.pop('ss58_address')
        if crypto_type != None:
            obj['crypto_type'] = crypto_type
        return  Key(**obj)

    def generate_mnemonic(self, words: int = 24) -> str:
        """
        params:
            words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24
        """
        mnemonic =  bip39_generate(words, self.language_code)
        assert bip39_validate(mnemonic, self.language_code), """Invalid mnemonic, please provide a valid mnemonic"""
        return mnemonic
        
    def from_mnemonic(self, mnemonic: str = None, crypto_type=crypto_type) -> 'Key':
        """
        Create a Key for given memonic
        """

        crypto_type = self.get_crypto_type(crypto_type)
        mnemonic = mnemonic or self.generate_mnemonic()
        if crypto_type == "ecdsa":
            if self.language_code != "en":
                raise ValueError("ECDSA mnemonic only supports english")
            peivate_key = mnemonic_to_ecdsa_private_key(mnemonic)
            keypair = self.from_private_key(mnemonic_to_ecdsa_private_key(mnemonic), crypto_type=crypto_type)
        else:
            seed_hex = binascii.hexlify(bytearray(bip39_to_mini_secret(mnemonic, "", self.language_code))).decode("ascii")
            if type(seed_hex) is str:
                seed_hex = bytes.fromhex(seed_hex.replace('0x', ''))
            if crypto_type == 'sr25519':
                public_key, private_key = sr25519.pair_from_seed(seed_hex)
            elif crypto_type == "ed25519":
                private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
            else:
                raise ValueError('crypto_type "{}" not supported'.format(crypto_type))
            ss58_address = ss58_encode(public_key, self.ss58_format)
            keypair = Key(private_key=private_key, crypto_type=crypto_type)
        keypair.mnemonic = mnemonic
        return keypair
   
    def from_private_key(
            self, 
            private_key: Union[bytes, str],
            crypto_type: int = crypto_type
    ) -> 'Key':
        """
        Creates Key for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        crypto_type: Use KeyType.[SR25519|ED25519|ECDSA] cryptography for generating the Key
        Returns
        -------
        Key
        """
        return Key(private_key=private_key, crypto_type=crypto_type)

    def encode_signature_data(self, data: Union[ScaleBytes, bytes, str, dict]) -> bytes:
        """
        Encodes data for signing and vefiying,  converting it to bytes if necessary.
        """
        data = copy.deepcopy(data)

        if not isinstance(data, str):
            data = python2str(data)
        if isinstance(data, str):
            if data[0:2] == '0x': # hex string
                data = bytes.fromhex(data[2:])
            elif type(data) is str:
                data = data.encode()
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        return data

    def resolve_signature(self, signature: Union[bytes, str]):
        if isinstance(signature,str) and signature[0:2] == '0x':
            signature = bytes.fromhex(signature[2:])
        if type(signature) is str:
            signature = bytes.fromhex(signature)
        if type(signature) is not bytes:
            raise TypeError(f"Signature should be of type bytes or a hex-string {signature}")
        return signature

    def resolve_public_key(self, address=None, public_key=None):
        if address != None:
            if is_valid_ss58_address(address):
                public_key = ss58_decode(address)
            else:
                public_key = address
        if public_key == None:
            public_key = self.public_key
        if isinstance(public_key, str) :
            if public_key.startswith('0x'):
                public_key = public_key[2:]
            public_key = bytes.fromhex(public_key)
        return public_key


    def get_sign_function(self, crypto_type=None):
        """
        Returns the sign function for the given crypto type
        """
        crypto_type = self.get_crypto_type(crypto_type)
        if crypto_type == "sr25519":
            return sr25519.sign
        elif crypto_type == "ed25519":
            return ed25519_zebra.ed_sign
        elif crypto_type == "ecdsa":
            return ecdsa_sign
        else:
            raise ValueError(f"Invalid crypto type: {crypto_type}")

        

    def sign(self, data: Union[ScaleBytes, bytes, str], mode='bytes') -> bytes:
        """
        Creates a signature for given data
        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format
        Returns
        -------
        signature in bytes

        """
        data = self.encode_signature_data(data)
        crypto_type = self.get_crypto_type(self.crypto_type)

        if crypto_type == "sr25519":
            signature = sr25519.sign((self.public_key, self.private_key), data)
        elif crypto_type == "ed25519":
            signature = ed25519_zebra.ed_sign(self.private_key, data)
        elif crypto_type == "ecdsa":
            signature = ecdsa_sign(self.private_key, data)
        else:
            raise Exception("Crypto type not supported")

        if mode in ['str', 'hex']:
            signature = '0x' + signature.hex()
        elif mode in ['dict', 'json']:
            signature =  {
                    'data':data.decode(),
                    'crypto_type':crypto_type,
                    'signature':signature.hex(),
                    'address': self.key_address}
        elif mode == 'bytes':
            signature = signature
        else:
            raise ValueError(f'invalid mode {mode}')

        return signature

    def verify(self, 
               data: Union[ScaleBytes, bytes, str, dict], 
               signature: Union[bytes, str] = None,
               address = None,
               public_key:Optional[str]= None, 
               max_age = None,
               crypto_type = None,
               **kwargs
               ) -> bool:
        """
        Verifies data with specified signature
        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format
        public_key: public key in bytes or hex string format
        """
        if isinstance(data, dict) and  all(k in data for k in ['data','signature', 'address']):
            data, signature, address = data['data'], data['signature'], data['address']
        data = self.encode_signature_data(data)
        signature = self.resolve_signature(signature)
        public_key = self.resolve_public_key(address=address, public_key=public_key)
        crypto_type = self.get_crypto_type(crypto_type)

        if crypto_type == "sr25519":
            crypto_verify_fn = sr25519.verify
        elif crypto_type == "ed25519":
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif crypto_type == "ecdsa":
            crypto_verify_fn = ecdsa_verify
        else:
            raise Exception("Crypto type not supported")
        verified = crypto_verify_fn(signature, data, public_key)
        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', public_key)
        return verified

    def encrypt(self, data, password=None, key=None):
        password = self.get_password(password=password, key=key)  
        data = copy.deepcopy(data)
        if not isinstance(data, str):
            data = str(data)
        data = data + (AES.block_size - len(data) % AES.block_size) * chr(AES.block_size - len(data) % AES.block_size)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(password, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        return encrypted_bytes.decode() 

    def decrypt(self, data, password=None, key=None):  
        password = self.get_password(password=password, key=key)  
        data = base64.b64decode(data)
        iv = data[:AES.block_size]
        cipher = AES.new(password, AES.MODE_CBC, iv)
        data =  cipher.decrypt(data[AES.block_size:])
        data = data[:-ord(data[len(data)-1:])].decode('utf-8')
        return data

    def get_password(self,  password:str=None, key:Optional[str]=None,):
        if password == None:
            password = (self if key == None else self.get_key(key)).private_key
        if isinstance(password, str):
            password = password.encode()
        # if password is a key, use the key's private key as password
        return hashlib.sha256(password).digest()

    def encrypt_key(self, path = 'test.enc', key=None, crypto_type=None,  password=None):
        assert self.key_exists(path), f'file {path} does not exist'
        assert not self.is_key_encrypted(path), f'{path} already encrypted'
        path = self.get_key_path(path)
        data = c.get(path)
        key = self.get_key(key)
        enc_data = key.encrypt(deepcopy(data), password=password)
        enc_text = {'data': enc_data, 
                    "key_address": data['key_address'],
                    "crypto_type": data['crypto_type'],
                    'encrypted': True}
        c.put(path, enc_text)
        return {'number_of_characters_encrypted':len(enc_text), 'path':path }
    
    def is_key_encrypted(self, key, data=None, crypto_type=None):
        return self.is_encrypted(self.get_key_data(key, crypto_type=crypto_type) )
    
    def decrypt_key(self, path = 'test.enc', crypto_type=None , password=None, key=None):
        crypto_type = self.get_crypto_type(crypto_type)
        assert self.key_exists(path, crypto_type=crypto_type), f'file {path} does not exist'
        assert self.is_key_encrypted(path, crypto_type=crypto_type), f'{path} not encrypted'
        path = self.get_key_path(path, crypto_type=crypto_type)
        data = self.get_key_data(path, crypto_type=crypto_type)
        assert self.is_encrypted(data), f'{path} not encrypted'
        key = self.get_key(key, crypto_type=crypto_type)
        dec_text =  key.decrypt(data['data'], password=password)
        c.put(path, dec_text)
        assert not self.is_key_encrypted(path, crypto_type=crypto_type ), f'failed to decrypt {path}'
        loaded_key = self.get_key(path, crypto_type=crypto_type)
        return { 'path':path , 'key_address': loaded_key.ss58_address,'crypto_type': loaded_key.crypto_type}

    def __str__(self):
        crypto_type = self.get_crypto_type(self.crypto_type)
        return  f'<Key(address={self.key_address} crypto_type={crypto_type}>'
        
    def is_encrypted(self, data):
        if isinstance(data, str):
            if data.startswith('{') and data.endswith('}'):
                try:
                    data  = json.loads(data)
                except:
                    pass
            if data in self.keys():
                data = self.get_key_data(data)
        return isinstance(data, dict) and bool(data.get('encrypted', False))

    def from_uri(
            self, 
            suri: str, 
            crypto_type=None, 
            DEV_PHRASE = 'bottom drive obey lake curtain smoke basket hold race lonely fit walk'

    ) -> 'Key':
        """
        Creates Key for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`

        Parameters
        ----------
        suri:
        crypto_type: Use "sr25519" or "ed25519"cryptography for generating the Key

        Returns
        -------
        Key
        """
        crypto_type = self.get_crypto_type(crypto_type)
        # GET THE MNEMONIC (PHRASE) AND DERIVATION PATHS
        suri = str(suri)
        if not suri.startswith('//'):
            suri = '//' + suri
        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri
        suri_parts = re.match(r'^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri).groupdict()
        mnemonic = suri_parts['phrase']
        crypto_type = self.get_crypto_type(crypto_type)
        if crypto_type == "ecdsa":
            private_key = mnemonic_to_ecdsa_private_key(
                mnemonic=mnemonic,
                str_derivation_path=suri_parts['path'],
                passphrase=suri_parts['password']
            )
            derived_keypair = self.from_private_key(private_key, crypto_type=crypto_type)
        elif crypto_type in ["sr25519", "ed25519"]:
            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for crypto_type '{crypto_type}'")
            derived_keypair = self.from_mnemonic(mnemonic, crypto_type=crypto_type)
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))
        return derived_keypair

    def from_password(self, password:str, crypto_type=None, **kwargs):
        return self.from_uri(password, crypto_type=crypto_type, **kwargs)

    def str2key(self, password:str, crypto_type=None, **kwargs):
        return self.from_password(password, crypto_type=crypto_type, **kwargs)

    def multi(self,key=None, crypto_type=None):
        key = self.get_key(key, crypto_type=crypto_type ) if key != None else self
        return key.crypto_type_name + '::' + key.key_address
    