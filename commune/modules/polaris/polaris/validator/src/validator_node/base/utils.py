import re
from dataclasses import asdict, dataclass

from communex.client import CommuneClient
from communex.compat.key import check_ss58_address
from communex.module.client import ModuleClient
from communex.types import Ss58Address
from loguru import logger
from substrateinterface import Keypair

IP_REGEX = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+")


def extract_address(string: str):
    """
    Extracts an address from a string.
    """
    return re.search(IP_REGEX, string)


def get_netuid(client: CommuneClient, subnet_name: str = "compute"):
    subnets = client.query_map_subnet_names()
    logger.info(f"Available subnets: {subnets}")
    
    for netuid, name in subnets.items():
        if name.lower() == subnet_name.lower():
            logger.info(f"Found netuid: {netuid} for subnet: {name}")
            return netuid
    
    logger.error(f"Subnet '{subnet_name}' not found. Available subnets: {list(subnets.values())}")
    raise ValueError(f"Subnet {subnet_name} not found")


def get_ip_port(modules_addresses: dict[int, str]):
    filtered_addr = {
        id: extract_address(addr) for id, addr in modules_addresses.items()
    }
    ip_port = {
        id: x.group(0).split(":") for id, x in filtered_addr.items() if x is not None
    }
    return ip_port
