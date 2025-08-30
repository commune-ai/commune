import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TypeGuard

import torustrateinterface.utils.ss58 as ss58
from nacl.exceptions import CryptoError
from pydantic import BaseModel
from torustrateinterface import Keypair

from torusdk._common import SS58_FORMAT
from torusdk.encryption import (
    decrypt_data,
    encrypt_data,
)
from torusdk.errors import PasswordNotProvidedError
from torusdk.password import NoPassword, PasswordProvider
from torusdk.types.types import Ss58Address
from torusdk.util import bytes_to_hex

TORUS_KEY_VERSION = 1
TORUS_HOME = "~/.torus"


class TorusKey(BaseModel):
    crypto_type: int
    seed_hex: str | None
    derive_path: str | None
    path: str
    public_key: str
    ss58_format: int
    ss58_address: Ss58Address
    private_key: str
    mnemonic: str | None


@dataclass
class EncryptionMetadata:
    kdf: str
    cipher: str
    cipher_text: str
    nonce_size: int


class TorusStorage(TorusKey):
    version: int
    encrypted: bool
    timestamp: int
    mnemonic_present: bool
    encryption_metadata: EncryptionMetadata | None


def local_key_adresses(
    password_provider: PasswordProvider = NoPassword(),
) -> dict[str, Ss58Address]:
    """
    Retrieves a mapping of local key names to their SS58 addresses.
    If password is passed, it will be used to decrypt every key.
    If password is not passed and ctx is,
    the user will be prompted for the password.
    """

    # TODO: refactor to return mapping of (key_name -> Keypair)
    # Outside of this, Keypair can be mapped to Ss58Address

    key_dir = os.path.expanduser(os.path.join(TORUS_HOME, "key"))
    key_dir = Path(key_dir)

    if not key_dir.exists():
        return {}

    key_names = [
        f.stem
        for f in key_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]

    addresses_map: dict[str, Ss58Address] = {}

    for key_name in key_names:
        # issue #12 https://github.com/agicommies/torus/issues/12
        # added check for key2address to stop error
        # from being thrown by wrong key type.
        if key_name == "key2address":
            print(
                "key2address is saved in an invalid format. It will be ignored."
            )
            continue

        password = password_provider.get_password(key_name)
        try:
            keypair = load_key(key_name, password=password)
        except PasswordNotProvidedError:
            password = password_provider.ask_password(key_name)
            keypair = load_key(key_name, password=password)

        addresses_map[key_name] = check_ss58_address(keypair.ss58_address)

    return addresses_map


def is_ss58_address(
    address: str, ss58_format: int = SS58_FORMAT
) -> TypeGuard[Ss58Address]:
    """
    Validates whether the given string is a valid SS58 address.

    Args:
        address: The string to validate.
        ss58_format: The SS58 format code to validate against.

    Returns:
        True if the address is valid, False otherwise.
    """

    # ? Weird type error
    return ss58.is_valid_ss58_address(address, valid_ss58_format=ss58_format)  # type: ignore


def check_ss58_address(
    address: str | Ss58Address, ss58_format: int = 42
) -> Ss58Address:
    """
    Validates whether the given string is a valid SS58 address.

    Args:
        address: The string to validate.
        ss58_format: The SS58 format code to validate against.

    Returns:
        The validated SS58 address.

    Raises:
        AssertionError: If the address is invalid.
    """

    assert is_ss58_address(
        address, ss58_format
    ), f"Invalid SS58 address '{address}'"
    return Ss58Address(address)


def generate_keypair() -> Keypair:
    """
    Generates a new keypair.
    """
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    return keypair


def key_path(name: str) -> str:
    """
    Constructs the file path for a key name.
    """

    home = Path.home()
    root_path = home / ".torus" / "key"
    name = name + ".json"
    return str(root_path / name)


def from_pydantic(data: TorusKey) -> Keypair:
    ss58_address = data.ss58_address
    private_key = data.private_key
    mnemonic_key = data.mnemonic
    public_key = data.public_key
    ss58_format = data.ss58_format
    if mnemonic_key:
        key = Keypair.create_from_mnemonic(mnemonic_key, ss58_format)
    else:
        key = Keypair.create_from_private_key(
            private_key, public_key, ss58_address, ss58_format
        )

    return key


def to_pydantic(keypair: Keypair, path: str) -> TorusKey:
    return TorusKey(
        path=path,
        mnemonic=keypair.mnemonic,
        public_key=bytes_to_hex(keypair.public_key),
        private_key=bytes_to_hex(keypair.private_key),
        ss58_address=check_ss58_address(keypair.ss58_address),
        seed_hex=bytes_to_hex(keypair.seed_hex) if keypair.seed_hex else None,
        ss58_format=keypair.ss58_format,
        crypto_type=keypair.crypto_type,
        derive_path=keypair.derive_path,
    )


def load_keypair(
    name: str,
    password: str | None = None,
    password_provider: PasswordProvider = NoPassword(),
) -> Keypair:
    """
    Loads a key from the filesystem.

    Args:
        name: The name of the key.
        password: The password to decrypt the key with.

    Returns:
        The loaded key.

    Raises:
        FileNotFoundError: If the key file does not exist.
        PasswordNotProvidedError: If the key is encrypted and no password is provided.
    """
    stored_key = load_key(name, password, password_provider)
    key = from_pydantic(stored_key)
    return key


def resolve_key_ss58(key: Ss58Address | Keypair | str) -> Ss58Address:
    """
    Resolves a keypair or key name to its corresponding SS58 address.

    If the input is already an SS58 address, it is returned as is.

    """

    if isinstance(key, Keypair):
        return key.ss58_address  # type: ignore

    if is_ss58_address(key):
        return key

    try:
        storage_obj = load_key_public(key)
        return storage_obj.ss58_address
    except FileNotFoundError:
        raise ValueError(
            f"Key is not a valid SS58 address nor a valid key name: {key}"
        )


def decrypt_storage(stored_key: TorusStorage, password: str) -> TorusStorage:
    if stored_key.seed_hex:
        stored_key.seed_hex = decrypt_data(password, stored_key.seed_hex)
    stored_key.private_key = decrypt_data(password, stored_key.private_key)
    if stored_key.mnemonic:
        stored_key.mnemonic = decrypt_data(password, stored_key.mnemonic)
    stored_key.encrypted = False
    return stored_key


def load_key(
    name: str,
    password: str | None = None,
    password_provider: PasswordProvider = NoPassword(),
):
    """
    Loads a key from the filesystem.

    Args:
        name: The name of the key.
        password: The password to decrypt the key with.

    Returns:
        The loaded key.

    Raises:
        FileNotFoundError: If the key file does not exist.
        PasswordNotProvidedError: If the key is encrypted and no password is provided.
    """
    path = key_path(name)
    full_path = os.path.expanduser(os.path.join(TORUS_HOME, path))
    try:
        with open(full_path, "r") as file:
            body = json.load(file)
        stored_key = TorusStorage.model_validate(body)
        if stored_key.encrypted:
            if password is None:
                password = password_provider.ask_password(name)
            stored_key = decrypt_storage(stored_key, password)
    except FileNotFoundError as err:
        raise FileNotFoundError(f"Key '{name}' not found", err)
    except CryptoError as err:
        raise CryptoError(f"Invalid password for key '{name}'", err)
    return stored_key


def load_key_public(name: str) -> TorusStorage:
    """
    Loads a key from the filesystem.

    Args:
        name: The name of the key.
        password: The password to decrypt the key with.

    Returns:
        The loaded key.

    Raises:
        FileNotFoundError: If the key file does not exist.
        PasswordNotProvidedError: If the key is encrypted and no password is provided.
    """
    path = key_path(name)
    full_path = os.path.expanduser(os.path.join(TORUS_HOME, path))
    with open(full_path, "r") as file:
        body = json.load(file)
    stored_key = TorusStorage.model_validate(body)
    return stored_key


def key_name_exists(name: str) -> bool:
    """
    Checks if a key with the given name exists.
    """
    path = key_path(name)
    return os.path.exists(path)


def store_key(keypair: Keypair, name: str, password: str | None = None) -> None:
    """
    Stores a key to the filesystem.

    Args:
        keypair: The key to store.
        name: The name of the key.
        password: The password to encrypt the key with.
    """
    path = key_path(name)
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    data = to_pydantic(keypair, name)
    is_mnemonic = keypair.mnemonic is not None
    if password is not None:
        if data.seed_hex:
            data.seed_hex, _ = encrypt_data(password, data.seed_hex)
        if data.mnemonic:
            data.mnemonic, _ = encrypt_data(password, data.mnemonic)
        data.private_key, nonce_size = encrypt_data(password, data.private_key)
        encrypted = True
        encryption_metadata = EncryptionMetadata(
            kdf="blake2b",
            cipher="xsalsa20-poly1305",
            cipher_text="base64",
            nonce_size=nonce_size,
        )
    else:
        encrypted = False
        encryption_metadata = None
    storage_obj = TorusStorage(
        version=1,
        encrypted=encrypted,
        mnemonic_present=is_mnemonic,
        timestamp=int(time()),
        encryption_metadata=encryption_metadata,
        **data.model_dump(),
    )
    with open(path, "w") as file:
        json.dump(storage_obj.model_dump(), file)
