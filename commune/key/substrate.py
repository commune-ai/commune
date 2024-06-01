
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
from substrateinterface.utils import ss58

from substrateinterface.constants import DEV_PHRASE
from substrateinterface.exceptions import ConfigurationError
from substrateinterface.key import extract_derive_path
from substrateinterface.utils.ecdsa_helpers import mnemonic_to_ecdsa_private_key, ecdsa_verify, ecdsa_sign
from substrateinterface.utils.encrypted_json import decode_pair_from_encrypted_json, encode_pair

from bip39 import bip39_to_mini_secret, bip39_generate, bip39_validate
import sr25519
import ed25519_zebra
import commune as c

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


class Keypair(c.Module):
    keys_path = c.data_path + '/keys.json'
    def __init__(self, 
                 ss58_address: str = None, 
                 public_key: Union[bytes, str] = None,
                 private_key: Union[bytes, str] = None, 
                 ss58_format: int = 42, 
                 seed_hex: Union[str, bytes] = None,
                 crypto_type: int = KeypairType.SR25519,
                 derive_path: str = None,
                 mnemonic: str = None,
                 path:str = None,
                 **kwargs
                 ):
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

        # If no arguments are provided, generate a random keypair
        if ss58_address == None \
            and public_key == None \
            and private_key == None \
            and seed_hex == None \
            and mnemonic == None:

            key = self.new_key()
            seed_hex = key.__dict__.get('seed_hex', seed_hex)
            private_key = key.__dict__.get('private_key', private_key)
            crypto_type = key.__dict__.get('crypto_type', crypto_type)
            derive_path = key.__dict__.get('derive_path', derive_path)
            ss58_address = key.__dict__.get('key', ss58_address)
            path = key.__dict__.get('path', path)
            public_key = key.__dict__.get('public_key', public_key)
            ss58_format = key.__dict__.get('ss58_format', ss58_format)
            mnemonic = key.__dict__.get('mnemonic', mnemonic)



        self.crypto_type = crypto_type
        self.seed_hex = seed_hex
        self.derive_path = None
        self.path = path 
        self.ss58_format = ss58_format


        if crypto_type != KeypairType.ECDSA and ss58_address and not public_key:
            public_key = ss58_decode(ss58_address, valid_ss58_format=ss58_format)

        if private_key:

            if type(private_key) == str:
                private_key = c.str2bytes(private_key)

            if self.crypto_type == KeypairType.SR25519:
                if len(private_key) != 64:
                    raise ValueError('Secret key should be 64 bytes long')
                if not public_key:
                    public_key = sr25519.public_from_secret_key(private_key)

            if self.crypto_type == KeypairType.ECDSA:
                private_key_obj = PrivateKey(private_key)
                public_key = private_key_obj.public_key.to_address()
                ss58_address = private_key_obj.public_key.to_checksum_address()
            
        if not public_key:
            raise ValueError('No SS58 formatted address or public key provided')

        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace('0x', ''))

        if crypto_type == KeypairType.ECDSA:
            if len(public_key) != 20:
                raise ValueError('Public key should be 20 bytes long')
        else:
            if len(public_key) != 32:
                raise ValueError('Public key should be 32 bytes long')

            if not ss58_address:
                ss58_address = ss58_encode(public_key, ss58_format=ss58_format)

        self.public_key: bytes = public_key

        self.ss58_address: str = ss58_address

        self.private_key: bytes = private_key

        self.mnemonic = mnemonic

    @classmethod
    def add_key(cls, path:str, mnemonic:str = None, password:str=None, refresh:bool=False, private_key=None, **kwargs):
        
        if cls.key_exists(path) and not refresh :
            c.print(f'key already exists at {path}', color='red')
            return json.loads(cls.get(path))
        c.print(f'generating key {path}')
        key = cls.new_key(mnemonic=mnemonic, private_key=private_key, **kwargs)
        key.path = path
        key_json = key.to_json()
        if password != None:
            key_json = cls.encrypt(data=key_json, password=password)
        cls.put(path, key_json)
        cls.update()
        return  json.loads(key_json)
    
    
    @classmethod
    def update(cls, **kwargs):
        return cls.key2address(update=True,**kwargs)
    
    @classmethod
    def rename_key(self, new_path):
        return self.mv_key(self.path, new_path)
    
    @classmethod
    def mv_key(cls, path, new_path):
        
        assert cls.key_exists(path), f'key does not exist at {path}'
        cls.put(new_path, cls.get_key(path).to_json())
        cls.rm_key(path)
        assert cls.key_exists(new_path), f'key does not exist at {new_path}'
        new_key = cls.get_key(new_path)
        return {'success': True, 'from': path , 'to': new_path, 'key': new_key}
    

    rename_key = mv_key 
    
    @classmethod
    def switch_keys(cls, path1:str, path2:str):
        
        assert path1 != path2
        assert cls.key_exists(path1), f'key does not exist at {path1}'
        assert cls.key_exists(path2), f'key does not exist at {path2}'

        before  = {
            path1: cls.key2address(path1),
            path2: cls.key2address(path2)
        }
        

        key1 = c.get_key(path1)
        key2 = c.get_key(path2)   
        cls.put(path1, key2.to_json()) 
        cls.put(path2, key1.to_json())


        after  = {
            path1 : cls.key2address(path1), 
            path2 : cls.key2address(path2)
        }

        assert before[path1] == after[path2]
        assert before[path2] == after[path1]
        
        return {'success': True, 'before': before, 'after': after, 'msg': f'switched {path1} and {path2}'}
    
    swap_keys = switch_keys
    @classmethod
    def add_keys(cls, name, n=100, verbose:bool = False, **kwargs):
        response = []
        for i in range(n):
            key_name = f'{name}.{i}'
            if bool == True:
                c.print(f'generating key {key_name}')
            response.append(cls.add_key(key_name, **kwargs))

        return response
    
    @classmethod
    def key_info(cls, path='module', create_if_not_exists=False, **kwargs):
        kwargs['json'] = True
        return cls.get_key(path, create_if_not_exists=create_if_not_exists, **kwargs)
    
    @classmethod
    def key_info_map(cls, *args, **kwargs):
        return {key: cls.key_info(key) for key in cls.keys(*args, **kwargs)}

    @classmethod
    def load_key(cls, path=None):
        key_info = cls.get(path)
        key_info = c.jload(key_info)
        if key_info['path'] == None:
            key_info['path'] = path.replace('.json', '').split('/')[-1]

        cls.add_key(**key_info)
        return {'status': 'success', 'message': f'key loaded from {path}'}
    

    @classmethod
    def load_keys(cls, path=keys_path, verbose:bool = False, refresh:bool = False,  **kwargs):
        return cls.load_mems(path, verbose=verbose, refresh=refresh, **kwargs)

    @classmethod
    def save_keys(cls, path='saved_keys.json', **kwargs):
        path = cls.resolve_path(path)
        c.print(f'saving mems to {path}')
        mems = cls.mems()
        c.put_json(path, mems)
        return {'saved_mems':list(mems.keys()), 'path':path}
    
    savemems = savekeys = save_keys 

    @classmethod
    def load_keys(cls, path='saved_keys.json', refresh=False, **kwargs):

        """"""
        mems = c.get_json(path)
        for k,mem in mems.items():
            try:
                cls.add_key(k, mnemonic=mem, refresh=refresh, **kwargs)
            except Exception as e:
                c.print(f'failed to load mem {k} due to {e}', color='red')
        return {'loaded_mems':list(mems.keys()), 'path':path}
    loadkeys = loadmems = load_keys
    

    @classmethod
    def mems(cls, search=None):
        mems = {}
        for key in cls.keys(search):
            try:
                mems[key] = cls.getmem(key)
            except Exception as e:
                c.print(f'failed to get mem for {key} due to {e}', color='red')
        if search:
            mems = {k:v for k,v in mems.items() if search in k or search in v}
        return mems

    mnemonics = mems
    

    @classmethod
    def get_key(cls, 
                path:str,
                password:str=None, 
                json:bool=False,
                create_if_not_exists:bool = False,
                **kwargs):
        if not cls.key_exists(path):
            if create_if_not_exists:
                key = cls.add_key(path, **kwargs)
                c.print(f'key does not exist, generating new key -> {key["key"]}')
            else:
                raise ValueError(f'key does not exist at --> {path}')
        
        key_json = cls.get(path)

        # if key is encrypted, decrypt it
        if cls.is_encrypted(key_json):
            key_json = c.decrypt(data=key_json, password=password)
            if key_json == None:
                c.print({'status': 'error', 'message': f'key is encrypted, please {path} provide password'}, color='red')
            return None


        if isinstance(key_json, str):
            key_json = c.jload(key_json)


        if json:
            key_json['path'] = path
            return key_json
        else:
            return cls.from_json(key_json)
        
        
        
    @classmethod
    def get_keys(cls, search=None, clean_failed_keys=False):
        keys = {}
        for key in cls.keys():
            if str(search) in key or search == None:
                try:
                    keys[key] = cls.get_key(key)
                except Exception as e:
                    c.print(f'failed to get key {key} due to {e}', color='red')
                    continue
                if keys[key] == None:
                    if clean_failed_keys:
                        cls.rm_key(key)
                    keys.pop(key)
                
        return keys
        

    @classmethod
    def key2address(cls, search=None, max_age=None, update=False, **kwargs):
        path = 'key2address'
        key2address = []
        key2address =  cls.get(path, key2address,max_age=max_age, update=update)
        if len(key2address) == 0:
            key2address =  { k: v.ss58_address for k,v  in cls.get_keys(search).items()}
            cls.put(path, key2address)
        if search != None:
            key2address =  {k:v for k,v in key2address.items() if  search in k}
        
        return key2address

    @classmethod
    def address2key(cls, search:Optional[str]=None, update:bool=False):
        address2key =  { v: k for k,v in cls.key2address(update=update).items()}
        if search != None :
            return address2key.get(search, None)
        return address2key
    
    @classmethod
    def get_address(cls, key):
        return cls.key2address()[key]
    get_addy = get_address
    @classmethod
    def has_address(cls, address):
        return address in cls.address2key()
    
    @classmethod
    def get_key_for_address(cls, address, ):
        return cls.address2key().get(address)

    key_storage_path = c.repo_path

    
    @classmethod
    def key_paths(cls):
        return cls.ls()
    
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
        key_exists =  key in cls.keys(**kwargs)
        if not key_exists:
            addresses = list(cls.key2address().values())
            if key in addresses:
                key_exists = True
        return key_exists
    

    @classmethod
    def get_key_address(cls, key): 
        return cls.get_key(key).ss58_address
    
    
    @classmethod
    def rm_key(cls, key=None):
        
        key2path = cls.key2path()
        keys = list(key2path.keys())
        if key not in keys:
            raise Exception(f'key {key} not found, available keys: {keys}')
        c.rm(key2path[key])
        cls.update()
        assert c.exists(key2path[key]) == False, 'key not deleted'

        return {'deleted':[key]}
    
    @property
    def crypto_type_name(self):
        return self.crypto_type2name(self.crypto_type)
        
    @classmethod
    def rm_keys(cls, rm_keys, verbose:bool=False):
        
        if isinstance(rm_keys, str):
            rm_keys = cls.keys(rm_keys)
        
        assert isinstance(rm_keys, list), f'rm_keys must be list, got {type(rm_keys)}'

        for rm_key in rm_keys:
            cls.rm_key(rm_key)
        
        return {'removed_keys':rm_keys}
    
    @classmethod
    def rm_all_keys(cls):
        return cls.rm_keys(cls.keys())
    
    crypto_types = ['ED25519', 'SR25519', 'ECDSA']

    @classmethod
    def crypto_type_map(cls):
        crypto_type_map =  {k:v for k,v in KeypairType.__dict__.items() if k in cls.crypto_types }
        return crypto_type_map

    @classmethod
    def crypto_name2type(cls, name:str):
        crypto_type_map = cls.crypto_type_map()
        name = name.upper()
        for k,v in crypto_type_map.items():
            if k.startswith(name.upper()):
                return v
        return crypto_type_map[name.upper()]
         
    @classmethod
    def crypto_type2name(cls, crypto_type:str):
        crypto_type_map ={v:k for k,v  in cls.crypto_type_map().items()}
        return crypto_type_map[crypto_type]
         
        
    @classmethod
    def resolve_crypto_type(cls, crypto_type):
            
        if isinstance(crypto_type, str):
            crypto_type = crypto_type.upper()
            crypto_type = cls.crypto_name2type(crypto_type)
        elif isinstance(crypto_type, int):
            assert crypto_type in list(KeypairType.__dict__.values()), f'crypto_type {crypto_type} not supported'
            
        assert crypto_type in list(KeypairType.__dict__.values()), f'crypto_type {crypto_type} not supported'
        return crypto_type
    
    @classmethod
    def new_key(cls, 
            mnemonic:str = None,
            suri:str = None, 
            private_key: str = None,
            crypto_type: Union[int,str] = 'sr25519', 
            json: bool = False,
            verbose:bool=False,
            **kwargs):
        '''
        yo rody, this is a class method you can gen keys whenever fam
        '''
        mnemonic = kwargs.pop('m', mnemonic)

        if verbose:
            c.print(f'generating {crypto_type} keypair, {suri}', color='green')

        crypto_type = cls.resolve_crypto_type(crypto_type)

        if suri:
            key =  cls.create_from_uri(suri, crypto_type=crypto_type)
        elif mnemonic:
            key = cls.create_from_mnemonic(mnemonic, crypto_type=crypto_type)
        elif private_key:
            key = cls.create_from_private_key(private_key,crypto_type=crypto_type)
        else:
            mnemonic = cls.generate_mnemonic()
            key = cls.create_from_mnemonic(mnemonic, crypto_type=crypto_type)
        if json:
            return key.to_json()
        
        return key
    
    create = gen = new_key

    
    
    def to_json(self, password: str = None ) -> dict:
        state_dict =  self.copy(self.__dict__)
        for k,v in state_dict.items():
            if type(v)  in [bytes]:
                state_dict[k] = v.hex() 
                if password != None:
                    state_dict[k] = self.encrypt(data=state_dict[k], password=password)
        if '_ss58_address' in state_dict:
            state_dict['key'] = state_dict.pop('_ss58_address')
        state_dict = json.dumps(state_dict)
        
        return state_dict
    
    @classmethod
    def from_json(cls, obj: Union[str, dict], password: str = None) -> dict:
        if type(obj) == str:
            obj = json.loads(obj)
        if obj == None:
           return None 
        
        for k,v in obj.items():
            if cls.is_encrypted(obj[k]) and password != None:
                obj[k] = cls.decrypt(data=obj[k], password=password)
        if 'key' in obj:
            obj['_ss58_address'] = obj.pop('key')
        return  cls(**obj)
    
    @classmethod
    def sand(cls):
        
        for k in cls.new_key(suri=2):
            
            password = 'fam'
            enc = cls.encrypt(k, password=password)
            dec = cls.decrypt(enc, password='bro ')
            
            



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


    # def resolve_crypto_type()
    @classmethod
    def create_from_mnemonic(cls, mnemonic: str = None, ss58_format=42, crypto_type=KeypairType.SR25519,
                             language_code: str = MnemonicLanguageCode.ENGLISH, return_kwargs:bool = False) -> 'Keypair':
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
        if not mnemonic:
            mnemonic = cls.generate_mnemonic(language_code=language_code)

        if crypto_type == KeypairType.ECDSA:
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
                return_kwargs=return_kwargs
            )

            if return_kwargs:
                kwargs = keypair
                return kwargs


        keypair.mnemonic = mnemonic



        return keypair

    from_mnemonic = from_mem = create_from_mnemonic

    @classmethod
    def create_from_seed(
            cls, 
            seed_hex: Union[bytes, str],
            ss58_format: Optional[int] = 42, 
            crypto_type=KeypairType.SR25519,
            return_kwargs:bool = False
            
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


        kwargs =  dict(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type, seed_hex=seed_hex
        )
        
        if return_kwargs:
            return kwargs 
        else:
            return cls(**kwargs)
    @classmethod
    def from_password(cls, password:str, **kwargs):
        return cls.create_from_uri(password, **kwargs)
    
    pwd2key = password2key = from_password


    @classmethod
    def from_uri(
            cls, 
            suri: str, 
            ss58_format: Optional[int] = 42, 
            crypto_type=KeypairType.SR25519, 
            language_code: str = MnemonicLanguageCode.ENGLISH
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
        crypto_type = cls.resolve_crypto_type(crypto_type)
        suri = str(suri)
        if not suri.startswith('//'):
            suri = '//' + suri

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
            derived_keypair = cls.create_from_private_key(private_key, ss58_format=ss58_format, crypto_type=crypto_type)
        else:

            if suri_parts['password']:
                raise NotImplementedError(f"Passwords in suri not supported for crypto_type '{crypto_type}'")

            derived_keypair = cls.create_from_mnemonic(
                suri_parts['phrase'], ss58_format=ss58_format, crypto_type=crypto_type, language_code=language_code
            )

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

                derived_keypair = Keypair(public_key=child_pubkey, private_key=child_privkey, ss58_format=ss58_format)

        return derived_keypair
    create_from_uri = from_uri
    from_mnem = from_mnemonic = create_from_mnemonic
    @classmethod
    def create_from_private_key(
            cls, private_key: Union[bytes, str], public_key: Union[bytes, str] = None, ss58_address: str = None,
            ss58_format: int = 42, crypto_type: int = KeypairType.SR25519
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
    from_private_key = create_from_private_key


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

    def export_to_encrypted_json(self, passphrase: str, name: str = None, path=None) -> dict:
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
    
    seperator = "<DATA::SIGNATURE>"
    
    def sign(self, 
             data: Union[ScaleBytes, bytes, str], 
             return_json:bool=False,
             return_string = False,
             seperator = seperator
             ) -> bytes:
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
            data = c.python2str(data)
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
        
        if return_json:
            return {
                'data': data.decode(),
                'crypto_type': self.crypto_type,
                'signature': signature.hex(),
                'address': self.ss58_address,
            }

        if return_string:
            return f'{data.decode()}{seperator}{signature.hex()}'
        return signature
    
    def ticket2address(self, ticket, **kwargs):
        return self.verify(ticket, **kwargs)

    def signature2address(self, sig, **kwargs):
        return self.verify(sig, return_address=True, **kwargs)
    sig2addy = signature2address

    
    def verify(self, 
               data: Union[ScaleBytes, bytes, str, dict], 
               signature: Union[bytes, str] = None,
               public_key:Optional[str]= None, 
               return_address = False,
               seperator = seperator,
               ss58_format = 42,
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
        True if data is signed with this Keypair, otherwise False
        """
        if isinstance(data, str) and seperator in data:
            data, signature = data.split(seperator)

        if max_age != None:
            staleness = c.timestamp() - int(data)
            assert staleness < max_age, f'data is too old, {staleness} seconds old, max_age is {max_age}'

        data = c.copy(data)
        
        if isinstance(data, dict):

            signature = data.pop('signature')
            public_key = c.ss58_decode(data['address'])
            if 'data' in data:
                data = data.pop('data')
            
            if not isinstance(data, str):
                data = c.python2str(data)

        if address != None:
            public_key = c.ss58_decode(address)
        if public_key == None:
            public_key = public_key or self.public_key
        else:
            if self.is_ss58(public_key):
                public_key = c.ss58_decode(public_key)

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

        if self.crypto_type == KeypairType.SR25519:
            crypto_verify_fn = sr25519.verify
        elif self.crypto_type == KeypairType.ED25519:
            crypto_verify_fn = ed25519_zebra.ed_verify
        elif self.crypto_type == KeypairType.ECDSA:
            crypto_verify_fn = ecdsa_verify
        else:
            raise ConfigurationError("Crypto type not supported")

        verified = crypto_verify_fn(signature, data, public_key)

        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(signature, b'<Bytes>' + data + b'</Bytes>', public_key)

        if return_address:
            return ss58_encode(public_key, ss58_format=ss58_format)
        return verified



        
        

    @property
    def encryption_key(self):
        password = None
        for k in ['private_key', 'mnemonic', 'sed_hex']:
            if hasattr(self, k):
                v = getattr(self, k)
                if type(v)  in [bytes]:
                    v = v.hex() 
                assert type(v) is str, f"Encryption key should be a string, not {type(v)}"
                
        assert password is not None, "No encryption key found, please make sure you have set either private_key, mnemonic or seed_hex"
        
        return password
    


    @property
    def aes_key(self):
        if not hasattr(self, '_aes_key'):
            password = self.mnemonic or self.private_key
            self._aes_key = c.module('key.aes')(c.bytes2str(password))
        return self._aes_key

    def encrypt(self, data: Union[str, bytes], password: str = None, **kwargs) -> bytes:
        aes_key = self.resolve_aes_key(password)
        return aes_key.encrypt(data, **kwargs)

    def resolve_aes_key(self, password = None):
        if password == None:
            key = Keypair.from_password(password)
        else: 
            key = self
        return key.aes_key

    def decrypt(self, data: Union[str, bytes], password=None, **kwargs) -> bytes:
        aes_key = self.resolve_aes_key(password)
        data = aes_key.decrypt(data)
        return data

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

    @classmethod
    def sandbox(cls ):
        key = cls.create_from_uri('//Alice')
        c.print(c.module('bittensor').get_balance(key.ss58_address))
        
    @classmethod
    def test(cls):
        self = cls.create_from_uri('//Alice')
        module_fns = c.fns()
        test_fns = [fn for fn in dir(self) if fn.startswith('test_') and fn not in module_fns ]
        num_tests = len(test_fns)
        results = {}
        for i, fn in enumerate(test_fns):
            try:
                result =  getattr(self, fn)()
            except Exception as e:
                result = c.detailed_error(e)
                c.print(f'Failed ({i+1}/{num_tests}) {fn} due to {e}', color='red')
            results[fn] = result
        return {'success':True, 'msg': 'all tests passed', 'results':results}
    @classmethod
    def is_key(cls, key) -> bool:
        return isinstance(key, Keypair)

    encrypted_prefix = 'ENCRYPTED::'

    @classmethod
    def encrypt_key(cls, path = 'test.enc', password=None):
        assert cls.exists(path), f'file {path} does not exist'
        if password == None:
            password = cls.generate_mnemonic()
        data = cls.get(path)
        enc_text =  c.encrypt(data, password=password)
        enc_text = f'{cls.encrypted_prefix}{enc_text}'
        cls.put(path, enc_text)
        return {'encrypted':enc_text, 'path':path , 'password':password}
    

    @classmethod
    def is_key_encrypted(cls, path, data=None):
        data = data or cls.get(path)
        if not isinstance(data, str):
            return False
        return data.startswith(cls.encrypted_prefix)
    
    
    @classmethod
    def test_key_encryption(cls, password='1234'):
        path = 'test.enc'
        c.add_key('test.enc', refresh=True)
        assert cls.is_key_encrypted(path) == False, f'file {path} is encrypted'
        cls.encrypt_key(path, password=password)
        assert cls.is_key_encrypted(path) == True, f'file {path} is not encrypted'
        cls.decrypt_key(path, password=password)
        assert cls.is_key_encrypted(path) == False, f'file {path} is encrypted'
        cls.rm(path)
        assert not c.exists(path), f'file {path} not deleted'
        return {'success': True, 'msg': 'test_key_encryption passed'}

    @classmethod
    def decrypt_key(cls, path = 'test.enc', password=None):
    
        data = cls.get(path)
        assert data.startswith(cls.encrypted_prefix), f'file {path} is not encrypted'
        data = data[len(cls.encrypted_prefix):]
        enc_text =  c.decrypt(data, password=password)
        cls.put(path, enc_text)
        return {'encrypted':enc_text, 'path':path , 'password':password}



    def encrypt_file(self, path='test.encryption', password=None, prefix=encrypted_prefix):
        if password == None:
            password = self.private_key
        text = c.get_text(path)
        enc_text =  self.encrypt(text, password=password)
        enc_text = f'{prefix}{enc_text}'
        c.put_text(path, enc_text)
        return {'encrypted':enc_text, 'path':path }




    def decrypt_file(self, path, password=None):
        if password == None:
            password = self.private_key
        enc_text = c.get_text(path)
        assert enc_text.startswith(self.encrypted_prefix), f'file {path} is not encrypted'
        enc_text = enc_text[len(self.encrypted_prefix):]
        dec_text =  self.decrypt(enc_text, password=password)
        if not isinstance(dec_text, str):
            dec_text = json.dumps(dec_text)
        c.put_text(path, dec_text)
        
        return {'success': True }


    
    def test_encryption_file(self, filepath='tests/dummy', value='test'):
        filepath = self.resolve_path(filepath)      
        c.put(filepath, value)
        decode = c.get(filepath)
        self.encrypt_file(filepath) # encrypt file
        decode = self.decrypt_file(filepath) # decrypt file
        decode = c.get(filepath)
        
        assert decode == value, f'encryption failed, {decode} != {value}'
        c.rm(filepath)
        assert not c.exists(filepath), f'file {filepath} not deleted'
        return {'success': True,
                'filepath': filepath,
                
                'msg': 'test_encryption_file passed'}
    
    
    @classmethod
    def test_encryption(cls,value = 10):
        key = cls.new_key()
        enc = key.encrypt(value)
        dec = key.decrypt(enc)
        assert dec == value, f'encryption failed, {dec} != {value}'
        return {'encrypted':enc, 'decrypted': dec}


    def test_key_encryption(self, test_key='test.key'):
        key = self.add_key(test_key, refresh=True)
        og_key = self.get_key(test_key)
        r = self.encrypt_key(test_key)
        self.decrypt_key(test_key, password=r['password'])
        key = self.get_key(test_key)

        assert key.ss58_address == og_key.ss58_address, f'key encryption failed, {key.ss58_address} != {self.ss58_address}'

        return {'success': True, 'msg': 'test_key_encryption passed'}
        
        


    def test_key_management(self):
        if self.key_exists('test'):
            self.rm_key('test')
        key1 = self.get_key('test')
        assert self.key_exists('test'), f'Key management failed, key still exists'
        self.mv_key('test', 'test2')
        key2 = self.get_key('test2')
        assert key1.ss58_address == key2.ss58_address, f'Key management failed, {key1.ss58_address} != {key2.ss58_address}'
        assert self.key_exists('test2'), f'Key management failed, key does not exist'
        assert not self.key_exists('test'), f'Key management failed, key still exists'
        self.mv_key('test2', 'test')
        assert self.key_exists('test'), f'Key management failed, key does not exist'
        assert not self.key_exists('test2'), f'Key management failed, key still exists'
        self.rm_key('test')
        assert not self.key_exists('test'), f'Key management failed, key still exists'
        return {'success': True, 'msg': 'test_key_management passed'}

    @classmethod
    def getmem(cls, key):
    
        return cls.get_key(key).mnemonic
    mem = getmem
    def __str__(self):
        return f'<Keypair (address={self.ss58_address}, path={self.path},  crypto_type: {self.crypto_type_name})>'


    def save(self, path=None):
        if path == None:
            path = self.path
        c.print(f'saving key to {path}')
        c.put_json(path, self.to_json())
        return {'saved':path}
    
    def diplicate(self, new_path):
        c.print(f'copying key from {self.path} to {new_path}')
        c.cp(self.path, new_path)
        return {'copied':new_path}
    
    

    def __repr__(self):
        return self.__str__()
        
    def state_dict(self):
        return self.__dict__
    
    to_dict = state_dict
    @classmethod
    def dashboard(cls): 
        import streamlit as st
        self = cls.new_key()
        
        
        keys = self.keys()
        
        selected_keys = st.multiselect('Keys', keys)
        buttons = {}
        for key_name in selected_keys:
            key = cls.get_key(key_name)
            with st.expander('Key Info'):
                st.write(key.to_dict())


            buttons[key_name] = {}
            buttons[key_name]['sign'] = st.button('Sign', key_name)
                
        st.write(self.keys())

    @classmethod
    def key2type(cls):
        keys = cls.keys(object=True)
        return {k.path: k.crypto_type_name for k in keys}
    @classmethod
    def key2mem(cls, search=None):
        keys = cls.keys(search, object=True)
        key2mem =  {k.path: k.mnemonic for k in keys}
        return key2mem
        
    @classmethod
    def type2keys(cls):
        type2keys = {}
        key2type = cls.key2type()
        for k,t in key2type.items():
            type2keys[t] = type2keys.get(t, []) + [k]
        return type2keys
        
    @classmethod
    def pubkey2multihash(cls, pk:bytes) -> str:
        import multihash
        hashed_public_key = multihash.encode(pk, code=multihash.SHA2_256)
        return hashed_public_key.hex()



    @classmethod
    def duplicate_keys(cls) -> dict:

        key2address = cls.key2address()
        duplicate_keys = {}

        for k,a in key2address.items():
            if a not in duplicate_keys:
                duplicate_keys[a] = []
        
            duplicate_keys[a] += [k]
        
        return {k:v for k,v in duplicate_keys.items() if len(v) > 1}

    @classmethod
    def clean_all_keys(cls):
        key2adress = c.key2address()
        for k,a in key2adress.items():
            if c.key_exists(a):
                c.print(f'removing {a}', color='red')
                c.rm_key(a)
            c.print('cleaning', k, a,  c.key_exists(a))

        
    @staticmethod
    def valid_ss58_address( address: str, valid_ss58_format:int=42  ) -> bool:
        """
        Checks if the given address is a valid ss58 address.

        Args:
            address(str): The address to check.

        Returns:
            True if the address is a valid ss58 address for Bittensor, False otherwise.
        """

        try:
            return ss58.valid_ss58_address( address, valid_ss58_format=valid_ss58_format ) # Default substrate ss58 format (legacy)
        except Exception as e:
            return False

    @classmethod
    def from_private_key(cls, private_key:str):
        return cls(private_key=private_key)
    
    @classmethod
    def valid_ss58_address(cls, address: str ) -> bool:
        """
        Checks if the given address is a valid ss58 address.

        Args:
            address(str): The address to check.

        Returns:
            True if the address is a valid ss58 address for Bittensor, False otherwise.
        """
        try:
            return ss58.is_valid_ss58_address( address, valid_ss58_format=c.__ss58_format__ )
        except (IndexError):
            return False
        
    @classmethod
    def is_valid_ed25519_pubkey(cls, public_key: Union[str, bytes] ) -> bool:
        """
        Checks if the given public_key is a valid ed25519 key.

        Args:
            public_key(Union[str, bytes]): The public_key to check.

        Returns:    
            True if the public_key is a valid ed25519 key, False otherwise.
        
        """
        try:
            if isinstance( public_key, str ):
                if len(public_key) != 64 and len(public_key) != 66:
                    raise ValueError( "a public_key should be 64 or 66 characters" )
            elif isinstance( public_key, bytes ):
                if len(public_key) != 32:
                    raise ValueError( "a public_key should be 32 bytes" )
            else:
                raise ValueError( "public_key must be a string or bytes" )

            keypair = Keypair(public_key=public_key,
                              ss58_format=c.__ss58_format__)

            ss58_addr = keypair.ss58_address
            return ss58_addr is not None

        except (ValueError, IndexError):
            return False

    @classmethod
    def is_valid_address_or_public_key(cls,  address: Union[str, bytes] ) -> bool:
        """
        Checks if the given address is a valid destination address.

        Args:
            address(Union[str, bytes]): The address to check.

        Returns:
            True if the address is a valid destination address, False otherwise.
        """
        if isinstance( address, str ):
            # Check if ed25519
            if address.startswith('0x'):
                return cls.is_valid_ed25519_pubkey( address )
            else:
                # Assume ss58 address
                return cls.valid_ss58_address( address )
        elif isinstance( address, bytes ):
            # Check if ed25519
            return cls.is_valid_ed25519_pubkey( address )
        else:
            # Invalid address type
            return False
        
    def id_card(self, return_json=True,**kwargs):
        return self.sign(str(c.timestamp()), return_json=return_json, **kwargs)

    def ticket(self, *args, **kwargs):
        return c.module('ticket')().ticket(*args,key=self.key, **kwargs)

    def verify_ticket(self, ticket, **kwargs):
        return c.module('ticket')().verify(ticket, key=self.key, **kwargs)
    
    def app(self):
        c.module('key.app').app()

    @staticmethod
    def is_ss58(address):
        # Check address length
        if len(address) != 47:
            return False
        
        # Check prefix
        network_prefixes = ['1', '2', '5', '7']  # Add more prefixes as needed
        if address[0] not in network_prefixes:
            return False
        
        # Verify checksum
        encoded = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        address_without_checksum = address[:-1]
        checksum = address[-1]
        address_hash = 0
        for char in address_without_checksum:
            address_hash = address_hash * 58 + encoded.index(char)
        
        # Calculate the expected checksum
        expected_checksum = encoded[address_hash % 58]
        
        # Compare the expected checksum with the provided checksum
        if expected_checksum != checksum:
            return False
        
        return True
 

    @classmethod
    def is_encrypted(cls, data, prefix=encrypted_prefix):
        if isinstance(data, str):
            if data.startswith(prefix):
                return True
        elif isinstance(data, dict):
            return bool(data.get('encrypted', False) == True)
        else:
            return False
    
    def test_signing(self):
        sig = self.sign('test')
        assert self.verify('test',sig, bytes.fromhex(self.public_key.hex()))
        assert self.verify('test',sig, self.public_key)
        sig = self.sign('test', return_string=True)
        assert self.verify(sig, self.public_key)
        return {'success':True}

    def test_move_key(self):
        self.add_key('testfrom')
        assert self.key_exists('testfrom')
        og_key = self.get_key('testfrom')
        self.mv_key('testfrom', 'testto')
        assert self.key_exists('testto')
        assert not self.key_exists('testfrom')
        new_key = self.get_key('testto')
        assert og_key.ss58_address == new_key.ss58_address
        self.rm_key('testto')
        assert not self.key_exists('testto')
        return {'success':True, 'msg':'test_move_key passed', 'key':new_key.ss58_address}

    def test_str_signing(self):
        sig = self.sign('test', return_string=True)
        # c.print(''+sig)
        assert not self.verify('1'+sig)
        assert self.verify(sig)
        return {'success':True}

Keypair.run(__name__)



