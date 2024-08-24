import random
import re
from collections import defaultdict
from enum import Enum
from typing import Mapping, TypeVar

from pydantic_settings import BaseSettings, SettingsConfigDict

from communex.balance import from_nano
from communex.types import Ss58Address

IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")


class ComxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COMX_")
    # TODO: improve node lists
    NODE_URLS: list[str] = [
        "wss://commune-api-node-0.communeai.net",
        "wss://commune-api-node-1.communeai.net",
        "wss://commune-api-node-2.communeai.net",
        "wss://commune-api-node-3.communeai.net",
        "wss://commune-api-node-4.communeai.net",
        "wss://commune-api-node-5.communeai.net",
        "wss://commune-api-node-6.communeai.net",
        "wss://commune-api-node-7.communeai.net",
        "wss://commune-api-node-8.communeai.net",
        "wss://commune-api-node-9.communeai.net",
        "wss://commune-api-node-10.communeai.net",
        "wss://commune-api-node-11.communeai.net",
        "wss://commune-api-node-12.communeai.net",
        "wss://commune-api-node-13.communeai.net",
        "wss://commune-api-node-14.communeai.net",
        "wss://commune-api-node-15.communeai.net",
        "wss://commune-api-node-16.communeai.net",
        "wss://commune-api-node-17.communeai.net",
        "wss://commune-api-node-18.communeai.net",
        "wss://commune-api-node-19.communeai.net",
        "wss://commune-api-node-20.communeai.net",
        "wss://commune-api-node-21.communeai.net",
        "wss://commune-api-node-22.communeai.net",
        "wss://commune-api-node-23.communeai.net",
        "wss://commune-api-node-24.communeai.net",
        "wss://commune-api-node-25.communeai.net",
        "wss://commune-api-node-26.communeai.net",
        "wss://commune-api-node-27.communeai.net",
        "wss://commune-api-node-28.communeai.net",
        "wss://commune-api-node-29.communeai.net",
        "wss://commune-api-node-30.communeai.net",
        "wss://commune-api-node-31.communeai.net",
    ]
    TESTNET_NODE_URLS: list[str] = ["wss://testnet-commune-api-node-0.communeai.net"]


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
            return f"{round_joules:,} J"


K = TypeVar("K")
V = TypeVar("V")
Z = TypeVar("Z")


def intersection_update(base: dict[K, V], update: dict[K, Z]) -> Mapping[K, V | Z]:
    """
    Update a dictionary with another dictionary, but only with keys that are already present.
    """
    updated = {k: update[k] for k in base if k in update}
    return updated


def transform_stake_dmap(stake_storage: dict[tuple[Ss58Address, Ss58Address], int]) -> dict[Ss58Address, list[tuple[Ss58Address, int]]]:
    """
    Transforms either the StakeTo or StakeFrom storage into the stake legacy data type.
    """
    transformed: dict[Ss58Address, list[tuple[Ss58Address, int]]] = defaultdict(list)
    [transformed[k1].append((k2, v)) for (k1, k2), v in stake_storage.items()]

    return dict(transformed)
