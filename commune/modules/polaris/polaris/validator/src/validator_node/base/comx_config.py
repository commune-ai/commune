import random
import re
import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Mapping, TypeVar

from communex.balance import from_nano
from communex.types import Ss58Address
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")


def deprecated(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"The function {func.__name__} is deprecated and may be removed in a future version.",
            DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


class ComxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMX_")
    # TODO: improve node lists
    NODE_URLS: list[str] = [
        "wss://api.communeai.net",
    ]
    TESTNET_NODE_URLS: list[str] = ["wss://testnet.api.communeai.net"]
    UNIVERSAL_PASSWORD: SecretStr | None = None
    KEY_PASSWORDS: dict[str, SecretStr] | None = None


def get_node_url(
    comx_settings: ComxSettings | None = None, *, use_testnet: bool = False
) -> str:
    comx_settings = comx_settings or ComxSettings()
    match use_testnet:
        case True:
            node_url = random.choice(comx_settings.TESTNET_NODE_URLS)
        case False:
            node_url = random.choice(comx_settings.NODE_URLS)
    return node_url


def get_available_nodes(
    comx_settings: ComxSettings | None = None, *, use_testnet: bool = False
) -> list[str]:
    comx_settings = comx_settings or ComxSettings()

    match use_testnet:
        case True:
            node_urls = comx_settings.TESTNET_NODE_URLS
        case False:
            node_urls = comx_settings.NODE_URLS
    return node_urls


class BalanceUnit(str, Enum):
    joule = "joule"
    j = "j"
    nano = "nano"
    n = "n"


def format_balance(balance: int, unit: BalanceUnit = BalanceUnit.nano) -> str:
    """
    Formats a balance.
    """

    match unit:
        case BalanceUnit.nano | BalanceUnit.n:
            return f"{balance}"
        case BalanceUnit.joule | BalanceUnit.j:
            in_joules = from_nano(balance)
            round_joules = round(in_joules, 4)
            return f"{round_joules:,} COMAI"


K = TypeVar("K")
V = TypeVar("V")
Z = TypeVar("Z")


def intersection_update(
    base: dict[K, V], update: dict[K, Z]
) -> Mapping[K, V | Z]:
    """
    Update a dictionary with another dictionary, but only with keys that are already present.
    """
    updated = {k: update[k] for k in base if k in update}
    return updated


def transform_stake_dmap(
    stake_storage: dict[tuple[Ss58Address, Ss58Address], int],
) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
    """
    Transforms either the StakeTo or StakeFrom storage into the stake legacy data type.
    """
    transformed: dict[Ss58Address, list[tuple[Ss58Address, int]]] = defaultdict(
        list
    )
    [transformed[k1].append((k2, v)) for (k1, k2), v in stake_storage.items()]

    return dict(transformed)
