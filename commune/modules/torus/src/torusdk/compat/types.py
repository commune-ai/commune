from typing import TypedDict

from torusdk.types.types import Ss58Address


class CommuneStorageDict(TypedDict):
    data: str
    encrypted: bool
    timestamp: int


class CommuneKeyDict(TypedDict):
    crypto_type: int
    seed_hex: str
    derive_path: str | None
    path: str
    public_key: str
    ss58_format: int
    ss58_address: Ss58Address
    private_key: str
    mnemonic: str
