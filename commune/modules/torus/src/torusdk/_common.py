import random
import re
import warnings
from collections import defaultdict
from typing import Any, Callable, Mapping, TypeVar

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from torusdk.types.types import Ss58Address

IPFS_REGEX = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$|bafk[1-7a-z]{52}$/i")
CID_REGEX = re.compile(
    r"^(?:ipfs://)?(?P<cid>Qm[1-9A-HJ-NP-Za-km-z]{44,}|b[A-Za-z2-7]{58,}|B[A-Z2-7]{58,}|z[1-9A-HJ-NP-Za-km-z]{48,}|F[0-9A-F]{50,})(?:/[\d\w.]+)*$"
)
SS58_FORMAT = 42


def deprecated(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"The function {func.__name__} is deprecated and may be removed in a future version.",
            DeprecationWarning,
        )
        return func(*args, **kwargs)

    return wrapper


class TorusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TORUS_")
    # TODO: improve node lists
    NODE_URLS: list[str] = [
        "wss://api.torus.network",
    ]
    TESTNET_NODE_URLS: list[str] = ["wss://api.testnet.torus.network"]
    UNIVERSAL_PASSWORD: SecretStr | None = None
    KEY_PASSWORDS: dict[str, SecretStr] | None = None


def get_node_url(
    torus_settings: TorusSettings | None = None, *, use_testnet: bool = False
) -> str:
    torus_settings = torus_settings or TorusSettings()
    match use_testnet:
        case True:
            node_url = random.choice(torus_settings.TESTNET_NODE_URLS)
        case False:
            node_url = random.choice(torus_settings.NODE_URLS)
    return node_url


def get_available_nodes(
    torus_settings: TorusSettings | None = None, *, use_testnet: bool = False
) -> list[str]:
    torus_settings = torus_settings or TorusSettings()

    match use_testnet:
        case True:
            node_urls = torus_settings.TESTNET_NODE_URLS
        case False:
            node_urls = torus_settings.NODE_URLS
    return node_urls


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
    Transforms either the StakingTo or StakedBy storage into the stake legacy data type.
    """
    transformed: dict[Ss58Address, list[tuple[Ss58Address, int]]] = defaultdict(
        list
    )
    [transformed[k1].append((k2, v)) for (k1, k2), v in stake_storage.items()]

    return dict(transformed)
