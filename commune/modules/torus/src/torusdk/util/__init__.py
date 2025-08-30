import ipaddress
import json
import os.path
import re
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

import requests


def check_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def ensure_dir_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir_exists(path: str) -> None:
    ensure_dir_exists(os.path.dirname(path))


def bytes_to_hex(value: str | bytes) -> str:
    """
    Converts a string or bytes object to its hexadecimal representation.

    If the input is already a string, it assumes that the string is already in
    hexadecimal format and returns it as is. If the input is bytes, it converts
    the bytes to their hexadecimal string representation.

    Args:
        x: The input string or bytes object to be converted to hexadecimal.

    Returns:
        The hexadecimal representation of the input.

    Examples:
        _to_hex(b'hello') returns '68656c6c6f'
        _to_hex('68656c6c6f') returns '68656c6c6f'
    """
    if isinstance(value, bytes):
        return value.hex()
    assert isinstance(value, str)
    # TODO: Check that `value` is a valid hexadecimal string
    return value


def is_ip_valid(ip: str) -> bool:
    """
    Checks if an ip address is valid
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


T = TypeVar("T")


class SetterGetterFn(Generic[T], Protocol):
    def __call__(self, x: T = ..., /) -> T: ...


def create_state_fn(default: Callable[..., T]) -> SetterGetterFn[T]:
    """
    Creates a state function that can be used to get or set a value.
    """
    value = default()

    def state_function(input: Optional[T] = None):
        nonlocal value
        if input is not None:
            value = input
        return value

    return state_function


def get_json_from_cid(cid: str) -> dict[Any, Any] | None:
    gateway = "https://ipfs.io/ipfs/"
    try:
        result = requests.get(gateway + cid)
        if result.ok:
            return result.json()
        return None
    except Exception:
        return None


def convert_cid_on_proposal(proposals: dict[int, dict[str, Any]]):
    unwrapped: dict[int, dict[str, Any]] = {}
    for prop_id, proposal in proposals.items():
        data = proposal.get("data")
        if data and "Custom" in data:
            metadata = proposal["metadata"]
            cid = metadata.split("ipfs://")[-1]
            queried_cid = get_json_from_cid(cid)
            if queried_cid:
                body = queried_cid.get("body")
                if body:
                    try:
                        queried_cid["body"] = json.loads(body)
                    except Exception:
                        pass
            proposal["Custom"] = queried_cid
        unwrapped[prop_id] = proposal
    return unwrapped


HEX_PATTERN = re.compile(r"^[0-9a-fA-F]+$")


# TODO: merge `is_hex_string` into `parse_hex`
def is_hex_string(string: str):
    return bool(HEX_PATTERN.match(string))


def parse_hex(hex_str: str) -> bytes:
    if hex_str[0:2] == "0x":
        return bytes.fromhex(hex_str[2:])
    else:
        return bytes.fromhex(hex_str)
