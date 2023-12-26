
import torch
import scalecodec
from retry import retry
from typing import List, Dict, Union, Optional, Tuple
from substrateinterface import SubstrateInterface
from typing import List, Dict, Union, Optional, Tuple
from commune.utils.network import ip_to_int, int_to_ip
from rich.prompt import Confirm
from commune.modules.subspace.balance import Balance
from commune.modules.subspace.utils import (U16_MAX,  is_valid_address_or_public_key, )
from commune.modules.subspace.chain_data import (ModuleInfo, custom_rpc_type_registry)

import streamlit as st
import json
from loguru import logger
import os
import commune as c

logger = logger.opt(colors=True)



class Subspace(c.Module):
    """
    Handles interactions with the subspace chain.
    """
    whitelist = ['modules']
    fmt = 'j'
    whitelist = []
    chain_name = 'subspace'
    git_url = 'https://github.com/commune-ai/subspace.git'
    default_config = c.get_config(chain_name, to_munch=False)
    token_decimals = default_config['token_decimals']
    network = default_config['network']
    chain = network
    libpath = chain_path = c.libpath + '/subspace'
    spec_path = f"{chain_path}/specs"
    netuid = default_config['netuid']
    image = 'vivonasg/subspace.libra:latest'
    mode = 'docker'
    telemetry_backend_image = 'parity/substrate-telemetry-backend'
    telemetry_frontend_image = 'parity/substrate-telemetry-frontend'
    node_key_prefix = 'subspace.node'


