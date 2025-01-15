import ed25519_zebra
import sr25519
import commune as c

from .types.index import *
from .utils import *

import binascii
import re
import nacl.public
from scalecodec.base import ScaleBytes
from bip39 import bip39_to_mini_secret
from scalecodec.utils.ss58 import (
    ss58_decode,
    ss58_encode
)
from typing import Union, Optional
from .key import Key, KeyType
class DotED25519(Key):
    def __init__(
        self,
        private_key: Union[bytes, str] = None,
        ss58_format: int = SS58_FORMAT,
        derive_path: str = None,
        path: str = None,
        crypto_type: Union[str, int] = KeyType.ED25519,
        **kwargs
    ):
        self.crypto_type = KeyType.ED25519
        self.set_private_key(private_key=private_key, ss58_format = ss58_format, derive_path=derive_path, path=path, **kwargs)
        
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
        # If no arguments are provided, generate a random keypair


        if private_key == None:
            private_key = self.new_key(crypto_type=self.crypto_type).private_key
        if type(private_key) == str:
            private_key = c.str2bytes(private_key)
        if self.crypto_type == KeyType.ED25519:
            private_key = private_key[:32] if len(private_key) == 64 else private_key        
            private_key, public_key = ed25519_zebra.ed_from_seed(private_key)
            key_address = ss58_encode(public_key, ss58_format=ss58_format)
            hash_type = 'ss58'
        else:
            raise ValueError('crypto_type "{}" not supported'.format(self.crypto_type))
        if type(public_key) is str:
            public_key = bytes.fromhex(public_key.replace("0x", ""))

        self.hash_type = hash_type
        self.public_key = public_key
        self.address = self.key_address = self.ss58_address = key_address
        self.private_key = private_key
        self.derive_path = derive_path
        self.path = path
        self.ss58_format = ss58_format
        self.key_address = self.ss58_address
        self.key_type = self.crypto_type2name(self.crypto_type)
        return {"key_address": key_address, "crypto_type": self.crypto_type}


    @classmethod
    def create_from_mnemonic(
        cls, mnemonic: str = None, ss58_format=SS58_FORMAT, language_code: str = "en", crypto_type=KeyType.ED25519
    ) -> "DotED25519":
        """
        Create a Key for given memonic

        Parameters
        ----------
        mnemonic: Seed phrase
        ss58_format: Substrate address format
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `"en"`

        Returns
        -------
        Key
        """
        if not mnemonic:
            mnemonic = cls.generate_mnemonic(language_code=language_code)
        
        keypair = cls.create_from_seed(
            seed_hex=binascii.hexlify(
                bytearray(bip39_to_mini_secret(mnemonic, "", language_code))
            ).decode("ascii"),
            ss58_format=ss58_format,
        )

        keypair.mnemonic = mnemonic

        return keypair

    @classmethod
    def create_from_seed(
        cls, seed_hex: Union[bytes, str], ss58_format: Optional[int] = SS58_FORMAT
    ) -> "DotED25519":
        """
        Create a Key for given seed

        Parameters
        ----------
        seed_hex: hex string of seed
        ss58_format: Substrate address format

        Returns
        -------
        Key
        """
        if type(seed_hex) is str:
            seed_hex = bytes.fromhex(seed_hex.replace("0x", ""))
        private_key, public_key = ed25519_zebra.ed_from_seed(seed_hex)
        ss58_address = ss58_encode(public_key, ss58_format)
        kwargs = dict(
            ss58_address=ss58_address,
            public_key=public_key,
            private_key=private_key,
            ss58_format=ss58_format,
            crypto_type=KeyType.ED25519,
        )

        return cls(**kwargs)

    @classmethod
    def create_from_uri(
        cls,
        suri: str,
        ss58_format: Optional[int] = SS58_FORMAT,
        language_code: str = "en",
        crypto_type: Union[str, int] = KeyType.ED25519,
    ) -> "DotED25519":
        """
        Creates Key for specified suri in following format: `[mnemonic]/[soft-path]//[hard-path]`

        Parameters
        ----------
        suri:
        ss58_format: Substrate address format
        language_code: The language to use, valid values are: 'en', 'zh-hans', 'zh-hant', 'fr', 'it', 'ja', 'ko', 'es'. Defaults to `"en"`

        Returns
        -------
        DotED25519
        """
        crypto_type = cls.resolve_crypto_type(crypto_type)
        suri = str(suri)
        if not suri.startswith("//"):
            suri = "//" + suri

        if suri and suri.startswith("/"):
            suri = DEV_PHRASE + suri

        suri_regex = re.match(
            r"^(?P<phrase>.[^/]+( .[^/]+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$",
            suri,
        )

        suri_parts = suri_regex.groupdict()

        if suri_parts["password"]:
            raise NotImplementedError(
                f"Passwords in suri not supported for crypto_type '{KeyType.ED25519}'"
            )

        derived_keypair = cls.create_from_mnemonic(
            suri_parts["phrase"], ss58_format=ss58_format, language_code=language_code
        )

        if suri_parts["path"] != "":
            derived_keypair.derive_path = suri_parts["path"]

            derive_junctions = extract_derive_path(suri_parts["path"])

            child_pubkey = derived_keypair.public_key
            child_privkey = derived_keypair.private_key

            for junction in derive_junctions:
                if junction.is_hard:
                    _, child_pubkey, child_privkey = sr25519.hard_derive_keypair(
                        (junction.chain_code, child_pubkey, child_privkey), b""
                    )

                else:
                    _, child_pubkey, child_privkey = sr25519.derive_keypair(
                        (junction.chain_code, child_pubkey, child_privkey), b""
                    )

            derived_keypair = DotED25519(
                public_key=child_pubkey,
                private_key=child_privkey,
                ss58_format=ss58_format,
            )

        return derived_keypair

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
        raise NotImplementedError(f"Cannot create JSON for crypto_type '{self.crypto_type}'")

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
        if not isinstance(data, str):
            data = c.python2str(data)
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == "0x":
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()
        if not self.private_key:
            raise Exception("No private key set to create signatures")
        signature = ed25519_zebra.ed_sign(self.private_key, data)

        if to_json:
            return {
                "data": data.decode(),
                "crypto_type": self.crypto_type,
                "signature": signature.hex(),
                "address": self.ss58_address,
            }
        return signature

    def verify(
        self,
        data: Union[ScaleBytes, bytes, str, dict],
        signature: Union[bytes, str] = None,
        public_key: Optional[str] = None,
        return_address=False,
        ss58_format=SS58_FORMAT,
        max_age=None,
        address=None,
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
        data = c.copy(data)

        if isinstance(data, dict):
            if self.is_ticket(data):
                address = data.pop("address")
                signature = data.pop("signature")
            elif "data" in data and "signature" in data and "address" in data:
                signature = data.pop("signature")
                address = data.pop("address", address)
                data = data.pop("data")
            else:
                assert signature != None, "signature not found in data"
                assert address != None, "address not found in data"

        if max_age != None:
            if isinstance(data, int):
                staleness = c.timestamp() - int(data)
            elif "timestamp" in data or "time" in data:
                timestamp = data.get("timestamp", data.get("time"))
                staleness = c.timestamp() - int(timestamp)
            else:
                raise ValueError(
                    "data should be a timestamp or a dict with a timestamp key"
                )
            assert staleness < max_age, (
                f"data is too old, {staleness} seconds old, max_age is {max_age}"
            )

        if not isinstance(data, str):
            data = c.python2str(data)
        if address != None:
            if self.valid_ss58_address(address):
                public_key = ss58_decode(address)
        if public_key == None:
            public_key = self.public_key
        if isinstance(public_key, str):
            public_key = bytes.fromhex(public_key.replace("0x", ""))
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == "0x":
            data = bytes.fromhex(data[2:])
        elif type(data) is str:
            data = data.encode()
        if type(signature) is str and signature[0:2] == "0x":
            signature = bytes.fromhex(signature[2:])
        elif type(signature) is str:
            signature = bytes.fromhex(signature)
        if type(signature) is not bytes:
            raise TypeError("Signature should be of type bytes or a hex-string")

        crypto_verify_fn = ed25519_zebra.ed_verify

        verified = crypto_verify_fn(signature, data, public_key)
        if not verified:
            # Another attempt with the data wrapped, as discussed in https://github.com/polkadot-js/extension/pull/743
            # Note: As Python apps are trusted sources on its own, no need to wrap data when signing from this lib
            verified = crypto_verify_fn(
                signature, b"<Bytes>" + data + b"</Bytes>", public_key
            )
        if return_address:
            return ss58_encode(public_key, ss58_format=ss58_format)
        return verified
