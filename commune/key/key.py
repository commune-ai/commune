import json
from typing import Union, Optional
import time
import os
import binascii
import re
import secrets
import base64
from base64 import b64encode
import hashlib
from Crypto import Random
from Crypto.Cipher import AES
import nacl.bindings
import nacl.public
from scalecodec.utils.ss58 import ss58_encode, ss58_decode, get_ss58_format
from scalecodec.base import ScaleBytes
from bip39 import bip39_generate, bip39_validate
import commune as c

from scalecodec.utils.ss58 import is_valid_ss58_address

from commune.key.constants import *
from .utils import *

class KeyType:
    ED25519 = 0
    SR25519 = 1
    ECDSA = 2
    SOLANA = 3
KeyType.crypto_types = [k for k in KeyType.__dict__.keys() if not k.startswith('_')]
KeyType.crypto_type_map =  {k.lower():v for k,v in KeyType.__dict__.items() if k in KeyType.crypto_types }
KeyType.crypto_types = list(KeyType.crypto_type_map.keys())

class Key(c.Module):
    crypto_types = KeyType.crypto_types
    crypto_type_map = KeyType.crypto_type_map
    crypto_types = list(crypto_type_map.keys())
    def __new__(
        cls,
        crypto_type: Union[str, int] = KeyType.SR25519,
        **kwargs,
    ):
        crypto_type = cls.resolve_crypto_type(crypto_type)
        if crypto_type == KeyType.SR25519:
            from .types.dot.sr25519 import DotSR25519
            return super().__new__(DotSR25519)
        elif crypto_type == KeyType.ED25519:
            from commune.key.types.dot.ed25519 import DotED25519
            return super().__new__(DotED25519)
        elif crypto_type == KeyType.ECDSA:
            from .types.eth import ECDSA
            return super().__new__(ECDSA)
        elif crypto_type == KeyType.SOLANA:
            from .types.sol import Solana
            return super().__new__(Solana)
        else:
            raise NotImplementedError(f"unsupported crypto_type {crypto_type}")
    
    @property
    def short_address(self):
        n = 4
        return self.ss58_address[:n] + "..." + self.ss58_address[-n:]

    def set_crypto_type(self, crypto_type):
        crypto_type = self.resolve_crypto_type(crypto_type)
        if crypto_type != self.crypto_type:
            kwargs = {
                "private_key": self.private_key,
                "ss58_format": self.ss58_format,
                "derive_path": self.derive_path,
                "path": self.path,
                "crypto_type": crypto_type,  # update crypto_type
            }
            return self.set_private_key(**kwargs)
        else:
            return {
                "success": False,
                "message": f"crypto_type already set to {crypto_type}",
            }

    def set_private_key(
        self,
        private_key: Union[bytes, str] = None,
        ss58_format: int = 42,
        derive_path: str = None,
        path: str = None,
        **kwargs,
    ):
        """
        Allows generation of Keys from a variety of input combination, such as a public/private key combination,
        mnemonic or URI containing soft and hard derivation paths. With these Keys data can be signed and verified

        Parameters
        ----------
        ss58_address: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        ss58_format: Substrate address format, default to 42 when omitted
        seed_hex: hex string of seed
        """
        raise NotImplementedError("set_private_key not implemented")

    @classmethod
    def add_key(
        cls,
        path: str,
        mnemonic: str = None,
        password: str = None,
        refresh: bool = False,
        private_key=None,
        crypto_type: Union[str, int] = KeyType.SR25519,
        **kwargs,
    ):
        if cls.key_exists(path) and not refresh:
            c.print(f"key already exists at {path}")
            return cls.get(path)
        key = cls.new_key(mnemonic=mnemonic, private_key=private_key, crypto_type=crypto_type, **kwargs)
        key.path = path
        key_json = key.to_json()
        if password != None:
            key_json = cls.encrypt(data=key_json, password=password)
        c.print(cls.put(path, key_json))
        cls.update()
        return json.loads(key_json)

    @classmethod
    def ticket(cls, data=None, key=None, **kwargs):
        return cls.get_key(key).sign(
            {"data": data, "time": c.time()}, to_json=True, **kwargs
        )

    @classmethod
    def mv_key(cls, path, new_path, crypto_type=KeyType.SR25519):
        assert cls.key_exists(path), f"key does not exist at {path}"
        cls.put(new_path, cls.get_key(path, crypto_type=crypto_type).to_json())
        cls.rm_key(path)
        assert cls.key_exists(new_path), f"key does not exist at {new_path}"
        assert not cls.key_exists(path), f"key still exists at {path}"
        new_key = cls.get_key(new_path, crypto_type=crypto_type)
        return {"success": True, "from": path, "to": new_path, "key": new_key}

    @classmethod
    def copy_key(cls, path, new_path):
        assert cls.key_exists(path), f"key does not exist at {path}"
        cls.put(new_path, cls.get_key(path).to_json())
        assert cls.key_exists(new_path), f"key does not exist at {new_path}"
        assert cls.get_key(path) == cls.get_key(new_path), "key does not match"
        new_key = cls.get_key(new_path)
        return {"success": True, "from": path, "to": new_path, "key": new_key}

    @classmethod
    def add_keys(cls, name, n=100, verbose: bool = False, **kwargs):
        response = []
        for i in range(n):
            key_name = f"{name}.{i}"
            if bool == True:
                c.print(f"generating key {key_name}")
            response.append(cls.add_key(key_name, **kwargs))

    def key2encrypted(self):
        keys = self.keys()
        key2encrypted = {}
        for k in keys:
            key2encrypted[k] = self.is_key_encrypted(k)
        return key2encrypted

    def encrypted_keys(self):
        return [k for k, v in self.key2encrypted().items() if v == True]

    @classmethod
    def key_info(cls, path="module", **kwargs):
        return cls.get_key_json(path)

    @classmethod
    def load_key(cls, path=None):
        key_info = cls.get(path)
        key_info = c.jload(key_info)
        if key_info["path"] == None:
            key_info["path"] = path.replace(".json", "").split("/")[-1]

        cls.add_key(**key_info)
        return {"status": "success", "message": f"key loaded from {path}"}

    @classmethod
    def save_keys(cls, path="saved_keys.json", **kwargs):
        path = cls.resolve_path(path)
        c.print(f"saving mems to {path}")
        key2mnemonic = cls.key2mnemonic()
        c.put_json(path, key2mnemonic)
        return {
            "success": True,
            "msg": "saved keys",
            "path": path,
            "n": len(key2mnemonic),
        }

    @classmethod
    def load_keys(cls, path="saved_keys.json", refresh=False, **kwargs):
        key2mnemonic = c.get_json(path)
        for k, mnemonic in key2mnemonic.items():
            try:
                cls.add_key(k, mnemonic=mnemonic, refresh=refresh, **kwargs)
            except Exception:
                # c.print(f'failed to load mem {k} due to {e}', color='red')
                pass
        return {"loaded_mems": list(key2mnemonic.keys()), "path": path}

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
                c.print(f"failed to get mem for {key} due to {e}")
        if search:
            mems = {k: v for k, v in mems.items() if search in k or search in v}
        return mems
    
    @classmethod
    def get_key(
        cls,
        path: str,
        crypto_type: Union[str, int] = KeyType.SR25519,
        password: str = None,
        create_if_not_exists: bool = True,
        **kwargs,
    ):
        crypto_type = cls.resolve_crypto_type(crypto_type)
        if hasattr(path, "key_address"):
            key = path
            return key
        path = path or "module"
        # if ss58_address is provided, get key from address
        if cls.valid_ss58_address(path):
            path = cls.address2key().get(path)
        if not cls.key_exists(path):
            if create_if_not_exists:
                key = cls.add_key(path, crypto_type=crypto_type, **kwargs)
                c.print(f"key does not exist, generating new key -> {key['path']}")
            else:
                print(path)
                raise ValueError(f"key does not exist at --> {path}")
        key_json = cls.get(path)
        # if key is encrypted, decrypt it
        if cls.is_encrypted(key_json):
            key_json = c.decrypt(data=key_json, password=password)
            if key_json == None:
                c.print(
                    {
                        "status": "error",
                        "message": f"key is encrypted, please {path} provide password",
                    }
                )
            return None
        key_json = c.jload(key_json) if isinstance(key_json, str) else key_json
        key = cls.from_json(key_json, crypto_type=crypto_type)
        key.path = path
        return key

    @classmethod
    def get_keys(cls, search=None, clean_failed_keys=False):
        keys = {}
        for key in cls.keys():
            if str(search) in key or search == None:
                try:
                    keys[key] = cls.get_key(key)
                except Exception:
                    continue
                if keys[key] == None:
                    if clean_failed_keys:
                        cls.rm_key(key)
                    keys.pop(key)
        return keys

    @classmethod
    def key2address(cls, search=None, max_age=10, update=False, **kwargs):
        path = "key2address"
        key2address = cls.get(path, None, max_age=max_age, update=update)
        if key2address == None:
            key2address = {k: v.ss58_address for k, v in cls.get_keys(search).items()}
            cls.put(path, key2address)
        return key2address

    @classmethod
    def n(cls, search=None, **kwargs):
        return len(cls.key2address(search, **kwargs))

    @classmethod
    def address2key(cls, search: Optional[str] = None, update: bool = False):
        address2key = {v: k for k, v in cls.key2address(update=update).items()}
        if search != None:
            return address2key.get(search, None)
        return address2key

    @classmethod
    def get_address(cls, key):
        return cls.get_key(key).ss58_address

    get_addy = get_address

    @classmethod
    def key_paths(cls):
        return cls.ls()

    address_seperator = "_address="

    @classmethod
    def key2path(cls) -> dict:
        """
        defines the path for each key
        """
        path2key_fn = lambda path: ".".join(path.split("/")[-1].split(".")[:-1])
        key2path = {path2key_fn(path): path for path in cls.key_paths()}
        return key2path

    @classmethod
    def keys(cls, search: str = None, **kwargs):
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
        import os

        return os.path.exists(path)

    @classmethod
    def get_key_path(cls, key):
        storage_dir = cls.storage_dir()
        key_path = storage_dir + "/" + key + ".json"
        return key_path

    @classmethod
    def get_key_json(cls, key):
        storage_dir = cls.storage_dir()
        key_path = storage_dir + "/" + key + ".json"
        return c.get(key_path)

    @classmethod
    def get_key_address(cls, key):
        return cls.get_key(key).ss58_address

    @classmethod
    def rm_key(cls, key=None):
        key2path = cls.key2path()
        keys = list(key2path.keys())
        if key not in keys:
            raise Exception(f"key {key} not found, available keys: {keys}")
        c.rm(key2path[key])
        return {"deleted": [key]}

    @classmethod
    def crypto_name2type(cls, name: str):
        crypto_type_map = cls.crypto_type_map
        name = name.lower()
        if name not in crypto_type_map:
            raise ValueError(f"crypto_type {name} not supported {crypto_type_map}")
        return crypto_type_map[name]

    @classmethod
    def crypto_type2name(cls, crypto_type: str):
        crypto_type_map = {v: k for k, v in cls.crypto_type_map.items()}
        return crypto_type_map[crypto_type]

    @classmethod
    def resolve_crypto_type_name(cls, crypto_type):
        return cls.crypto_type2name(cls.resolve_crypto_type(crypto_type))

    @classmethod
    def resolve_crypto_type(cls, crypto_type):
        if isinstance(crypto_type, int) or (
            isinstance(crypto_type, str) and c.is_int(crypto_type)
        ):
            crypto_type = int(crypto_type)
            crypto_type_map = cls.crypto_type_map
            reverse_crypto_type_map = {v: k for k, v in crypto_type_map.items()}
            assert crypto_type in reverse_crypto_type_map, (
                f"crypto_type {crypto_type} not supported {crypto_type_map}"
            )
            crypto_type = reverse_crypto_type_map[crypto_type]
        if isinstance(crypto_type, str):
            crypto_type = crypto_type.lower()
            crypto_type = cls.crypto_name2type(crypto_type)
        return int(crypto_type)

    @classmethod
    def new_private_key(cls):
        return cls.new_key().private_key.hex()

    @classmethod
    def new_key(
        cls,
        mnemonic: str = None,
        suri: str = None,
        private_key: str = None,
        verbose: bool = False,
        crypto_type: Union[str, int] = KeyType.SR25519,
        **kwargs,
    ):
        """
        yo rody, this is a class method you can gen keys whenever fam
        """
        if verbose:
            c.print(f"generating polkadot keypair, {suri}")

        if suri:
            key = cls.create_from_uri(suri, crypto_type=crypto_type)
        elif mnemonic:
            key = cls.create_from_mnemonic(mnemonic, crypto_type=crypto_type)
        elif private_key:
            key = cls.create_from_private_key(private_key, crypto_type=crypto_type)
        else:
            mnemonic = cls.generate_mnemonic()
            key = cls.create_from_mnemonic(mnemonic, crypto_type=crypto_type)
        return key

    create = gen = new_key

    def to_json(self, password: str = None) -> dict:
        state_dict = c.copy(self.__dict__)
        for k, v in state_dict.items():
            if type(v) in [bytes]:
                state_dict[k] = v.hex()
                if password != None:
                    state_dict[k] = self.encrypt(data=state_dict[k], password=password)
        if "_ss58_address" in state_dict:
            state_dict["ss58_address"] = state_dict.pop("_ss58_address")

        state_dict = json.dumps(state_dict)

        return state_dict

    @classmethod
    def from_json(cls, obj: Union[str, dict], password: str = None, crypto_type: Union[str, int] = KeyType.SR25519) -> dict:
        if type(obj) == str:
            obj = json.loads(obj)
        if obj == None:
            return None
        for k, v in obj.items():
            if cls.is_encrypted(obj[k]) and password != None:
                obj[k] = cls.decrypt(data=obj[k], password=password)
        if "ss58_address" in obj:
            obj["_ss58_address"] = obj.pop("ss58_address")
        obj["crypto_type"] = crypto_type
        return cls(**obj)
    
    @classmethod
    def generate_mnemonic(cls, words: int = 12, language_code: str = "en") -> str:
        """
        params:
            words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24
            language_code: The language to use, valid values are: 'en', 'zh-hans',
                           'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'.
                            Defaults to `"en"`
        """
        mnemonic = bip39_generate(words, language_code)
        assert cls.validate_mnemonic(mnemonic, language_code), "mnemonic is invalid"
        return mnemonic

    @classmethod
    def validate_mnemonic(cls, mnemonic: str, language_code: str = "en") -> bool:
        """
        Verify if specified mnemonic is valid

        Parameters
        ----------
        mnemonic: Seed phrase
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `"en"`

        Returns
        -------
        bool
        """
        return bip39_validate(mnemonic, language_code)

    @classmethod
    def create_from_mnemonic(
        cls, mnemonic: str, ss58_format: int = SS58_FORMAT, language_code: str = "en", crypto_type: Union[str, int] = KeyType.SR25519
    ) -> "Key":
        """
        Create a Key for given memonic

        Parameters
        ----------
        mnemonic: Seed phrase
        ss58_format: Substrate address format
        crypto_type: Use `KeyType.SR25519` or `KeyType.ED25519` cryptography for generating the Key
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `"en"`

        Returns
        -------
        Key
        """
        crypto_type = cls.resolve_crypto_type(crypto_type)
        if crypto_type == KeyType.SR25519:
            from commune.key.types.dot.sr25519 import DotSR25519
            return DotSR25519.create_from_mnemonic(mnemonic=mnemonic, ss58_format=ss58_format, language_code=language_code, crypto_type=crypto_type)
        elif crypto_type == KeyType.ED25519:
            from commune.key.types.dot.ed25519 import DotED25519
            return DotED25519.create_from_mnemonic(mnemonic=mnemonic, ss58_format=ss58_format, language_code=language_code, crypto_type=crypto_type)
        elif crypto_type == KeyType.ECDSA:
            from commune.key.types.eth import ECDSA
            return ECDSA.create_from_mnemonic(mnemonic=mnemonic, ss58_format=ss58_format, language_code=language_code, crypto_type=crypto_type)
        elif crypto_type == KeyType.SOLANA:
            from commune.key.types.sol import Solana
            return Solana.create_from_mnemonic(mnemonic=mnemonic, ss58_format=ss58_format, language_code=language_code, crypto_type=crypto_type)
        else:
            raise NotImplementedError("create_from_mnemonic not implemented")

    from_mnemonic = from_mem = create_from_mnemonic

    @classmethod
    def create_from_seed(
        cls,
        seed_hex: Union[bytes, str] = None,
        ss58_format: Optional[int] = SS58_FORMAT,
        crypto_type: Union[str, int] = KeyType.SR25519
    ) -> "Key":
        """
        Create a Key for given seed

        Parameters
        ----------
        seed_hex: hex string of seed
        ss58_format: Substrate address format
        crypto_type: Use KeyType.SR25519 or KeyType.ED25519 cryptography for generating the Key

        Returns
        -------
        Key
        """
        crypto_type = cls.resolve_crypto_type(crypto_type)
        if crypto_type == KeyType.SR25519:
            from commune.key.types.dot.sr25519 import DotSR25519
            return DotSR25519.create_from_seed(seed_hex=seed_hex, ss58_format=ss58_format, crypto_type=crypto_type)
        elif crypto_type == KeyType.ED25519:
            from commune.key.types.dot.ed25519 import DotED25519
            return DotED25519.create_from_seed(seed_hex=seed_hex, ss58_format=ss58_format, crypto_type=crypto_type)
        elif crypto_type == KeyType.ECDSA:
            from commune.key.types.eth import ECDSA
            return ECDSA.create_from_seed(seed_hex=seed_hex, ss58_format=ss58_format, crypto_type=crypto_type)
        elif crypto_type == KeyType.SOLANA:
            from commune.key.types.sol import Solana
            return Solana.create_from_seed(seed_hex=seed_hex, ss58_format=ss58_format, crypto_type=crypto_type)
        else:
            raise NotImplementedError("create_from_seed not implemented")

    @classmethod
    def create_from_password(cls, password: str, crypto_type: Union[str, int] = KeyType.SR25519, **kwargs):
        key = cls.create_from_uri(password, crypto_type=crypto_type, **kwargs)
        key.set_crypto_type(crypto_type)

    str2key = pwd2key = password2key = from_password = create_from_password

    @classmethod
    def create_from_uri(
        cls,
        suri: str,
        ss58_format: int = SS58_FORMAT,
        crypto_type: Union[str, int] = KeyType.SR25519,
        language_code: str = "en",
    ) -> "Key":
        """
        Creates Key for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`

        Parameters
        ----------
        suri:
        ss58_format: Substrate address format
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `"en"`

        Returns
        -------
        Key
        """
        if crypto_type == KeyType.SR25519:
            from commune.key.types.dot.sr25519 import DotSR25519
            return DotSR25519.create_from_uri(suri=suri, ss58_format=ss58_format, language_code=language_code)
        elif crypto_type == KeyType.ED25519:
            from commune.key.types.dot.ed25519 import DotED25519
            return DotED25519.create_from_uri(suri=suri, ss58_format=ss58_format, language_code=language_code)
        elif crypto_type == KeyType.ECDSA:
            from commune.key.types.eth import ECDSA
            return ECDSA.create_from_uri(suri=suri, ss58_format=ss58_format, language_code=language_code)
        elif crypto_type == KeyType.SOLANA:
            from commune.key.types.sol import Solana
            return Solana.create_from_uri(suri=suri, ss58_format=ss58_format, language_code=language_code)
        else:
            raise NotImplementedError("create_from_uri not implemented")


    @classmethod
    def create_from_private_key(
        cls,
        private_key: Union[bytes, str],
        public_key: Union[bytes, str] = None,
        ss58_address: str = None,
        ss58_format: int = SS58_FORMAT,
        crypto_type: Union[str, int] = KeyType.SR25519,
    ) -> "Key":
        """
        Creates Key for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        public_key: hex string or bytes of public key
        ss58_address: Substrate address
        ss58_format: Substrate address format, default = 42
        crypto_type: Use KeyType.[SR25519|ED25519|ECDSA] cryptography for generating the Key

        Returns
        -------
        Key
        """
        return cls(
            private_key=private_key,
            public_key=public_key,
            crypto_type=crypto_type,
            ss58_format=ss58_format,
            ss58_address=ss58_address
        )

    from_private_key = create_from_private_key

    @classmethod
    def create_from_encrypted_json(
        cls, json_data: Union[str, dict], passphrase: str, ss58_format: int = None
    ) -> "Key":
        """
        Create a Key from a PolkadotJS format encrypted JSON file

        Parameters
        ----------
        json_data: Dict or JSON string containing PolkadotJS export format
        passphrase: Used to encrypt the keypair
        ss58_format: Which network ID to use to format the SS58 address (42 for testnet)

        Returns
        -------
        Key
        """

        if type(json_data) is str:
            json_data = json.loads(json_data)
        private_key, public_key = decode_pair_from_encrypted_json(json_data, passphrase)
        if 'sr25519' in json_data['encoding']['content']:
            crypto_type = KeyType.SR25519
        elif 'ed25519' in json_data['encoding']['content']:
            crypto_type = KeyType.ED25519
            # Strip the nonce part of the private key
            private_key = private_key[0:32]
        elif 'solana' in json_data['encoding']['content']:
            crypto_type = KeyType.SOLANA
            private_key = private_key[0:32]
        else:
            raise NotImplementedError("Unknown KeyType found in JSON")
        
        if ss58_format is None and 'address' in json_data:
            ss58_format = get_ss58_format(json_data['address'])

        return cls.create_from_private_key(private_key, public_key, ss58_format=ss58_format, crypto_type=crypto_type)

    def export_to_encrypted_json(self, passphrase: str, name: str = None) -> dict:
        """
        Export Key to PolkadotJS format encrypted JSON file

        Parameters
        ----------
        passphrase: Used to encrypt the keypair
        name: Display name of Key used

        Returns
        -------
        dict
        """
        raise NotImplementedError("export_to_encrypted_json not implemented")

    seperator = "::signature="

    def sign(self, data: Union[ScaleBytes, bytes, str], to_json=False) -> bytes:
        """
        Creates a signature for given data
        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format
        Returns
        -------
        signature in bytes

        """
        raise NotImplementedError("sign not implemented")

    @classmethod
    def bytes2str(cls, data: bytes, mode: str = "utf-8") -> str:
        if hasattr(data, "hex"):
            return data.hex()
        else:
            if isinstance(data, str):
                return data
            return bytes.decode(data, mode)

    @classmethod
    def python2str(cls, input):
        from copy import deepcopy
        import json

        input = deepcopy(input)
        input_type = type(input)
        if input_type == str:
            return input
        if input_type in [dict]:
            input = json.dumps(input)
        elif input_type in [bytes]:
            input = cls.bytes2str(input)
        elif input_type in [list, tuple, set]:
            input = json.dumps(list(input))
        elif input_type in [int, float, bool]:
            input = str(input)
        return input

    def verify(
        self,
        data: Union[ScaleBytes, bytes, str, dict],
        signature: Union[bytes, str],
        public_key: Optional[str],
        return_address,
        ss58_format,
        max_age,
        address,
        **kwargs,
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
        raise NotImplementedError("verify not implemented")

    def is_ticket(self, data):
        return all(
            [k in data for k in ["data", "signature", "address", "crypto_type"]]
        ) and any([k in data for k in ["time", "timestamp"]])

    def resolve_encryption_password(self, password: str = None) -> str:
        if password == None:
            password = self.private_key
        if isinstance(password, str):
            password = password.encode()
        return hashlib.sha256(password).digest()

    def resolve_encryption_data(self, data):
        if not isinstance(data, str):
            data = str(data)
        return data

    def encrypt(self, data, password=None):
        data = self.resolve_encryption_data(data)
        password = self.resolve_encryption_password(password)
        data = data + (AES.block_size - len(data) % AES.block_size) * chr(
            AES.block_size - len(data) % AES.block_size
        )
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(password, AES.MODE_CBC, iv)
        encrypted_bytes = base64.b64encode(iv + cipher.encrypt(data.encode()))
        return encrypted_bytes.decode()

    def decrypt(self, data, password=None):
        password = self.resolve_encryption_password(password)
        data = base64.b64decode(data)
        iv = data[: AES.block_size]
        cipher = AES.new(password, AES.MODE_CBC, iv)
        data = cipher.decrypt(data[AES.block_size :])
        data = data[: -ord(data[len(data) - 1 :])].decode("utf-8")
        return data

    def encrypt_message(
        self,
        message: Union[bytes, str],
        recipient_public_key: bytes,
        nonce: bytes = secrets.token_bytes(24),
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
            raise Exception("No private key set to encrypt")
        if self.crypto_type != KeyType.ED25519:
            raise Exception("Only ed25519 keypair type supported")
        curve25519_public_key = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(
            recipient_public_key
        )
        recipient = nacl.public.PublicKey(curve25519_public_key)
        private_key = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(
            self.private_key + self.public_key
        )
        sender = nacl.public.PrivateKey(private_key)
        box = nacl.public.Box(sender, recipient)
        return box.encrypt(
            message if isinstance(message, bytes) else message.encode("utf-8"), nonce
        )

    def decrypt_message(
        self, encrypted_message_with_nonce: bytes, sender_public_key: bytes
    ) -> bytes:
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
            raise Exception("No private key set to decrypt")
        if self.crypto_type != KeyType.ED25519:
            raise Exception("Only ed25519 keypair type supported")
        private_key = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(
            self.private_key + self.public_key
        )
        recipient = nacl.public.PrivateKey(private_key)
        curve25519_public_key = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(
            sender_public_key
        )
        sender = nacl.public.PublicKey(curve25519_public_key)
        return nacl.public.Box(recipient, sender).decrypt(encrypted_message_with_nonce)

    encrypted_prefix = "ENCRYPTED::"

    @classmethod
    def encrypt_key(cls, path="test.enc", password=None):
        assert cls.key_exists(path), f"file {path} does not exist"
        assert not cls.is_key_encrypted(path), f"{path} already encrypted"
        data = cls.get(path)
        enc_text = {"data": c.encrypt(data, password=password), "encrypted": True}
        cls.put(path, enc_text)
        return {"number_of_characters_encrypted": len(enc_text), "path": path}

    @classmethod
    def is_key_encrypted(cls, key, data=None):
        data = data or cls.get(key)
        return cls.is_encrypted(data)

    @classmethod
    def decrypt_key(cls, path="test.enc", password=None, key=None):
        assert cls.key_exists(path), f"file {path} does not exist"
        assert cls.is_key_encrypted(path), f"{path} not encrypted"
        data = cls.get(path)
        assert cls.is_encrypted(data), f"{path} not encrypted"
        dec_text = c.decrypt(data["data"], password=password)
        cls.put(path, dec_text)
        assert not cls.is_key_encrypted(path), f"failed to decrypt {path}"
        loaded_key = c.get_key(path)
        return {
            "path": path,
            "key_address": loaded_key.ss58_address,
            "crypto_type": loaded_key.crypto_type,
        }

    @classmethod
    def get_mnemonic(cls, key):
        return cls.get_key(key).mnemonic

    def __str__(self):
        return (
            f"<Key(address={self.key_address} type={self.key_type} path={self.path})>"
        )

    def save(self, path=None):
        if path == None:
            path = self.path
        c.put_json(path, self.to_json())
        return {"saved": path}

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_private_key(cls, private_key: str):
        return cls(private_key=private_key)

    @classmethod
    def valid_ss58_address(cls, address: str, ss58_format:int = SS58_FORMAT) -> bool:
        """
        Checks if the given address is a valid ss58 address.
        """
        try:
            return is_valid_ss58_address(address, valid_ss58_format=ss58_format)
        except Exception:
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
            return bool(data.get("encrypted", False))
        else:
            return False

    @staticmethod
    def ss58_encode(*args, **kwargs):
        return ss58_encode(*args, **kwargs)

    @staticmethod
    def ss58_decode(*args, **kwargs):
        return ss58_decode(*args, **kwargs)

    @classmethod
    def resolve_key_address(cls, key):
        key2address = c.key2address()
        if key in key2address:
            address = key2address[key]
        else:
            address = key
        return address

    @classmethod
    def valid_h160_address(cls, address):
        # Check if it starts with '0x'
        if not address.startswith("0x"):
            return False

        # Remove '0x' prefix
        address = address[2:]

        # Check length
        if len(address) != 40:
            return False

        # Check if it contains only valid hex characters
        if not re.match("^[0-9a-fA-F]{40}$", address):
            return False

        return True

    def storage_migration(self):
        key2path = self.key2path()
        new_key2path = {}
        for k_name, k_path in key2path.items():
            try:
                key = c.get_key(k_name)
                new_k_path = (
                    "/".join(k_path.split("/")[:-1])
                    + "/"
                    + f"{k_name}_address={key.ss58_address}_type={key.crypto_type}.json"
                )
                new_key2path[k_name] = new_k_path
            except Exception as e:
                c.print(f"failed to migrate {k_name} due to {e}", color="red")

        return new_key2path

    def storage_migration(self):
        key2path = self.key2path()
        new_key2path = {}
        for k_name, k_path in key2path.items():
            try:
                key = c.get_key(k_name)
                new_k_path = (
                    "/".join(k_path.split("/")[:-1])
                    + "/"
                    + f"{k_name}_address={key.ss58_address}_type={key.crypto_type}.json"
                )
                new_key2path[k_name] = new_k_path
            except Exception as e:
                c.print(f"failed to migrate {k_name} due to {e}", color="red")

        return new_key2path