"""
Key storage compatible with the classic `commune` library.

WIP
"""

import json
import os
from pathlib import Path
from typing import Any, cast

from nacl.exceptions import CryptoError
from torustrateinterface import Keypair

from torusdk.compat.storage import COMMUNE_HOME, classic_load, classic_put
from torusdk.compat.types import CommuneKeyDict
from torusdk.errors import (
    InvalidPasswordError,
    KeyNotFoundError,
    PasswordNotProvidedError,
)
from torusdk.key import check_ss58_address, is_ss58_address, store_key
from torusdk.password import NoPassword, PasswordProvider
from torusdk.types.types import Ss58Address
from torusdk.util import bytes_to_hex, check_str


def check_key_dict(key_dict: Any) -> CommuneKeyDict:
    """
    Validates a given dictionary as a commune key dictionary and returns it.

    This function checks if the provided dictionary adheres to the structure of
    a CommuneKeyDict, that is used by the classic `commune` library and returns
    it if valid.

    Args:
        key_dict: The dictionary to validate.

    Returns:
        The validated commune key dictionary. Same as input.

    Raises:
      AssertionError: If the dictionary does not conform to the expected
        structure.
    """

    assert isinstance(key_dict, dict)
    assert isinstance(key_dict["crypto_type"], int)
    assert isinstance(key_dict["seed_hex"], str)
    assert isinstance(key_dict["derive_path"], str | None)
    assert isinstance(key_dict["path"], str) or key_dict["path"] is None
    assert isinstance(key_dict["public_key"], str)
    assert isinstance(key_dict["ss58_format"], int)
    assert isinstance(key_dict["ss58_address"], str)
    assert is_ss58_address(key_dict["ss58_address"])
    assert isinstance(key_dict["private_key"], str)
    assert isinstance(key_dict["mnemonic"], str)
    return cast(CommuneKeyDict, key_dict)


def classic_key_path(name: str) -> str:
    """
    Constructs the file path for a key name in the classic commune format.
    """

    home = Path.home()
    root_path = home / ".commune" / "key"
    name = name + ".json"
    return str(root_path / name)


def from_classic_dict(
    data: dict[Any, Any], from_mnemonic: bool = True
) -> Keypair:
    """
    Creates a `Key` from a dict conforming to the classic `commune` format.

    Args:
        data: The key data in a classic commune format.
        name: The name to assign to the key.

    Returns:
        The reconstructed `Key` instance.

    Raises:
        AssertionError: If `data` does not conform to the expected format.
    """

    data_ = check_key_dict(data)

    ss58_address = data_["ss58_address"]
    private_key = data_["private_key"]
    mnemonic_key = data_["mnemonic"]
    public_key = data_["public_key"]
    ss58_format = data_["ss58_format"]
    if from_mnemonic:
        key = Keypair.create_from_mnemonic(mnemonic_key, ss58_format)
    else:
        key = Keypair.create_from_private_key(
            private_key, public_key, ss58_address, ss58_format
        )

    return key


def to_classic_dict(keypair: Keypair, path: str) -> CommuneKeyDict:
    """
    Converts a keypair to a dictionary conforming to the classic commune format.

    Args:
        keypair: The keypair to convert.
        path: The path/name of the key file.
    """

    return {
        "path": path,
        "mnemonic": check_str(keypair.mnemonic),
        "public_key": bytes_to_hex(keypair.public_key),
        "private_key": bytes_to_hex(keypair.private_key),
        "ss58_address": check_ss58_address(keypair.ss58_address),
        "seed_hex": bytes_to_hex(keypair.seed_hex),
        "ss58_format": keypair.ss58_format,
        "crypto_type": keypair.crypto_type,
        "derive_path": keypair.derive_path,
    }


def classic_load_key(
    name: str,
    password: str | None = None,
    from_mnemonic: bool = True,  # Should be removed in future versions
) -> Keypair:
    """
    Loads the keypair with the given name from a disk.
    """
    path = classic_key_path(name)
    key_dict_json = classic_load(path, password=password)
    key_dict = json.loads(key_dict_json)
    try:
        kp = from_classic_dict(key_dict, from_mnemonic=from_mnemonic)
    except ValueError:
        kp = from_classic_dict(key_dict, from_mnemonic=(not from_mnemonic))
    return kp


def try_classic_load_key(
    key_name: str,
    password: str | None = None,
    *,
    password_provider: PasswordProvider = NoPassword(),
) -> Keypair:
    password = password or password_provider.get_password(key_name)
    try:
        try:
            keypair = classic_load_key(key_name, password=password)
        except PasswordNotProvidedError:
            password = password_provider.ask_password(key_name)
            keypair = classic_load_key(key_name, password=password)
    except FileNotFoundError as err:
        raise KeyNotFoundError(
            f"Key '{key_name}' is not a valid SS58 address nor a valid key name",
            err,
        )
    except CryptoError as err:
        raise InvalidPasswordError(
            f"Invalid password for key '{key_name}'", err
        )

    return keypair


def is_encrypted(name: str) -> bool:
    """
    Checks if the key with the given name is encrypted.
    """
    path = classic_key_path(name)
    full_path = os.path.expanduser(os.path.join(COMMUNE_HOME, path))
    with open(full_path, "r") as file:
        body = json.load(file)
    return body["encrypted"]


def classic_store_key(
    keypair: Keypair, name: str, password: str | None = None
) -> None:
    """
    Stores the given keypair on a disk under the given name.
    """
    key_dict = to_classic_dict(keypair, name)
    key_dict_json = json.dumps(key_dict)
    path = classic_key_path(name)
    classic_put(path, key_dict_json, password=password)


def legacy_resolve_key_ss58(key: Ss58Address | Keypair | str) -> Ss58Address:
    """
    Resolves a keypair or key name to its corresponding SS58 address.

    If the input is already an SS58 address, it is returned as is.

    DEPRECATED
    """

    if isinstance(key, Keypair):
        return key.ss58_address  # type: ignore

    if is_ss58_address(key):
        return key

    try:
        keypair = classic_load_key(key)
    except FileNotFoundError:
        raise ValueError(
            f"Key is not a valid SS58 address nor a valid key name: {key}"
        )

    address = keypair.ss58_address

    return check_ss58_address(address)


def migrate_to_torus(name: str, password_provider: PasswordProvider):
    password = None
    try:
        commune_key = classic_load_key(name)
    except PasswordNotProvidedError:
        password = password_provider.ask_password(name)
        commune_key = classic_load_key(name, password=password)
    store_key(commune_key, name, password)


def legacy_resolve_key_ss58_encrypted(
    key: Ss58Address | Keypair | str,
    password: str | None = None,
    password_provider: PasswordProvider = NoPassword(),
) -> Ss58Address:
    """
    Resolves a keypair or key name to its corresponding SS58 address.

    If the input is already an SS58 address, it is returned as is.
    """

    if isinstance(key, Keypair):
        return check_ss58_address(key.ss58_address, key.ss58_format)

    if is_ss58_address(key):
        return key

    keypair = try_classic_load_key(
        key, password=password, password_provider=password_provider
    )
    address = keypair.ss58_address
    return check_ss58_address(address, keypair.ss58_format)


def legacy_local_key_adresses(
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

    home = Path.home()
    key_dir = home / ".commune" / "key"

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
            keypair = classic_load_key(key_name, password=password)
        except PasswordNotProvidedError:
            password = password_provider.ask_password(key_name)
            keypair = classic_load_key(key_name, password=password)

        addresses_map[key_name] = check_ss58_address(keypair.ss58_address)

    return addresses_map
