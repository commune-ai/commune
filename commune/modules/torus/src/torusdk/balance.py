from enum import Enum
from typing import Any, TypeVar

DECIMALS = 18
UNIT_NAME = "Rems"


def from_rems(amount: int) -> float:
    """
    Converts from nano to j
    """

    return amount / (10**DECIMALS)


def to_rems(amount: float) -> int:
    """
    Converts from j to nano
    """

    return int(amount * (10**DECIMALS))


class BalanceUnit(str, Enum):
    joule = "rems"
    j = "r"
    nano = "torus"
    n = "t"


def format_balance(balance: int, unit: BalanceUnit = BalanceUnit.nano) -> str:
    """
    Formats a balance.
    """

    match unit:
        case BalanceUnit.nano | BalanceUnit.n:
            return f"{balance}"
        case BalanceUnit.joule | BalanceUnit.j:
            in_joules = from_rems(balance)
            round_joules = round(in_joules, 4)
            return f"{round_joules:,}" + " \u2653"


def from_horus(amount: int, subnet_tempo: int = 100) -> float:
    """
    Converts from horus to j
    """

    return amount / (10**DECIMALS * subnet_tempo)


def repr_j(amount: int):
    """
    Given an amount in nano, returns a representation of it in tokens/TOR.

    E.g. "103.2J".
    """

    return f"{from_rems(amount)} {UNIT_NAME}"


T = TypeVar("T", str, int)


def dict_from_nano(dict_data: dict[T, Any], fields_to_convert: list[str | int]):
    """
    Converts specified fields in a dictionary from nano to J. Only works for
    fields that are integers. Fields not found are silently ignored.
    Recursively searches nested dictionaries.
    """
    transformed_dict: dict[T, Any] = {}
    for key, value in dict_data.items():
        if isinstance(value, dict):
            transformed_dict[key] = dict_from_nano(value, fields_to_convert)  # type: ignore
        elif key in fields_to_convert:
            if not (isinstance(value, int) or value is None):
                raise ValueError(
                    f"Field {key} is not an integer in the dictionary."
                )
            transformed_dict[key] = repr_j(value)
        else:
            transformed_dict[key] = value

    return transformed_dict
