#
# Copyright 2022 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
#
# Copyright 2021 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
"""
This is copied from Web3 python library to control the `requests`
session parameters.
"""

import lru
import requests
from requests.adapters import HTTPAdapter
from web3._utils.caching import generate_cache_key


def _remove_session(key, session):
    session.close()


_session_cache = lru.LRU(8, callback=_remove_session)


def _get_session(*args, **kwargs):
    cache_key = generate_cache_key((args, kwargs))
    if cache_key not in _session_cache:
        # This is the main change from original Web3 `_get_session`
        session = requests.sessions.Session()
        session.mount(
            "http://",
            HTTPAdapter(pool_connections=25, pool_maxsize=25, pool_block=True),
        )
        session.mount(
            "https://",
            HTTPAdapter(pool_connections=25, pool_maxsize=25, pool_block=True),
        )
        _session_cache[cache_key] = session
    return _session_cache[cache_key]


def make_post_request(endpoint_uri, data, *args, **kwargs):
    kwargs.setdefault("timeout", 10)
    session = _get_session(endpoint_uri)
    response = session.post(endpoint_uri, data=data, *args, **kwargs)
    response.raise_for_status()

    return response.content




import os
from web3 import HTTPProvider, WebsocketProvider


class CustomHTTPProvider(HTTPProvider):
    """
    Override requests to control the connection pool to make it blocking.
    """

    def make_request(self, method, params):
        self.logger.debug(
            "Making request HTTP. URI: %s, Method: %s", self.endpoint_uri, method
        )
        request_data = self.encode_rpc_request(method, params)
        raw_response = make_post_request(
            self.endpoint_uri, request_data, **self.get_request_kwargs()
        )
        response = self.decode_rpc_response(raw_response)
        self.logger.debug(
            "Getting response HTTP. URI: %s, " "Method: %s, Response: %s",
            self.endpoint_uri,
            method,
            response,
        )
        return response


def get_web3_connection_provider(network_url):
    if network_url.startswith("http"):
        provider = CustomHTTPProvider(network_url)
    elif network_url.startswith("ws"):
        provider = WebsocketProvider(network_url)
    else:
        raise NotImplementedError
    return provider




from typing import Dict, Optional, Union
from web3 import WebsocketProvider
from web3.main import Web3


def get_web3(network_url: str) -> Web3:
    """
    Return a web3 instance connected via the given network_url.

    Adds POA middleware when connecting to the Rinkeby Testnet.

    A note about using the `rinkeby` testnet:
    Web3 py has an issue when making some requests to `rinkeby`
    - the issue is described here: https://github.com/ethereum/web3.py/issues/549
    - and the fix is here: https://web3py.readthedocs.io/en/latest/middleware.html#geth-style-proof-of-authority
    """
    from web3.middleware import geth_poa_middleware

    provider = get_web3_connection_provider(network_url)
    web3 = Web3(provider)

    if web3.eth.chain_id == 4:
        web3.middleware_onion.inject(geth_poa_middleware, layer=0)
    return web3

def get_web3_connection_provider(
    network_url: str,
) -> Union[CustomHTTPProvider, WebsocketProvider]:
    """Return the suitable web3 provider based on the network_url.

    Requires going through some gateway such as `infura`.

    Using infura has some issues if your code is relying on evm events.
    To use events with an infura connection you have to use the websocket interface.

    Make sure the `infura` url for websocket connection has the following format
    wss://rinkeby.infura.io/ws/v3/357f2fe737db4304bd2f7285c5602d0d
    Note the `/ws/` in the middle and the `wss` protocol in the beginning.

    :param network_url: str
    :return: provider : Union[CustomHTTPProvider, WebsocketProvider]
    """
    if network_url.startswith("http"):
        return CustomHTTPProvider(network_url)
    elif network_url.startswith("ws"):
        return WebsocketProvider(network_url)
    else:
        msg = (
            f"The given network_url *{network_url}* does not start with either"
            f"`http` or `wss`. A correct network url is required."
        )
        raise AssertionError(msg)
