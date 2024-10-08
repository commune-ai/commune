"""
Common types for the communex module.
"""
import json
from enum import Enum
from typing import NewType, TypedDict

Ss58Address = NewType("Ss58Address", str)
"""Substrate SS58 address.

The `SS58 encoded address format`_ is based on the Bitcoin Base58Check format,
but with a few modification specifically designed to suite Substrate-based
chains.

.. _SS58 encoded address format:
    https://docs.substrate.io/reference/address-formats/
"""

# TODO: replace with dataclasses

# == Burn related
MinBurn = NewType("MinBurn", int)
MaxBurn = NewType("MaxBurn", int)
BurnConfig = NewType("BurnConfig", dict[MinBurn, MaxBurn])


class VoteMode (Enum):
    authority = "Authority"
    vote = "Vote"


class GovernanceConfiguration(TypedDict):
    proposal_cost: int
    proposal_expiration: int
    vote_mode: int  # 0: Authority, 1: Vote
    proposal_reward_treasury_allocation: float
    max_proposal_reward_treasury_allocation: int
    proposal_reward_interval: int


class BurnConfiguration(TypedDict):
    min_burn: int
    max_burn: int
    adjustment_alpha: int
    target_registrations_interval: int
    target_registrations_per_interval: int
    max_registrations_per_interval: int


class NetworkParams(TypedDict):
    # max
    max_name_length: int
    min_name_length: int  # dont change the position
    max_allowed_subnets: int
    max_allowed_modules: int
    max_registrations_per_block: int
    max_allowed_weights: int

    # mins
    floor_delegation_fee: int
    floor_founder_share: int
    min_weight_stake: int

    # S0 governance
    curator: Ss58Address
    general_subnet_application_cost: int

    # Other
    subnet_immunity_period: int
    governance_config: GovernanceConfiguration

    kappa: int
    rho: int

class SubnetParamsMaps(TypedDict):
    netuid_to_founder: dict[int, Ss58Address]
    netuid_to_founder_share: dict[int, int]
    netuid_to_incentive_ratio: dict[int, int]
    netuid_to_max_allowed_uids: dict[int, int]
    netuid_to_max_allowed_weights: dict[int, int]
    netuid_to_min_allowed_weights: dict[int, int]
    netuid_to_max_weight_age: dict[int, int]
    netuid_to_name: dict[int, str]
    netuid_to_tempo: dict[int, int]
    netuid_to_trust_ratio: dict[int, int]
    netuid_to_bonds_ma: dict[int, int]
    netuid_to_maximum_set_weight_calls_per_epoch: dict[int, int]
    netuid_to_emission: dict[int, int]
    netuid_to_immunity_period: dict[int, int]
    netuid_to_governance_configuration: dict[int, GovernanceConfiguration]
    netuid_to_min_validator_stake: dict[int, int]
    netuid_to_max_allowed_validators: dict[int, int]
    netuid_to_module_burn_config: dict[int, BurnConfiguration]
    netuid_to_subnet_metadata: dict[int, str]


class SubnetParams(TypedDict):
    name: str
    tempo: int
    min_allowed_weights: int
    max_allowed_weights: int
    max_allowed_uids: int
    max_weight_age: int
    trust_ratio: int
    founder_share: int
    incentive_ratio: int
    founder: Ss58Address
    maximum_set_weight_calls_per_epoch: int | None
    bonds_ma: int | None
    immunity_period: int
    governance_config: GovernanceConfiguration
    min_validator_stake: int | None
    max_allowed_validators: int | None
    module_burn_config: BurnConfiguration
    subnet_metadata: str | None


# redundant "TypedDict" inheritance because of pdoc warns.
# see https://github.com/mitmproxy/pdoc/blob/26d40827ddbe1658e8ac46cd092f17a44cf0287b/pdoc/doc.py#L691-L692
class SubnetParamsWithEmission(SubnetParams, TypedDict):
    """SubnetParams with emission field."""

    emission: int
    """Subnet emission percentage (0-100).
    """


class ModuleInfo(TypedDict):
    uid: int
    key: Ss58Address
    name: str
    address: str  # "<ip>:<port>"
    emission: int
    incentive: int
    dividends: int
    stake_from: list[tuple[Ss58Address, int]]
    regblock: int  # block number
    last_update: int  # block number
    stake: int
    delegation_fee: int
    metadata: str | None


class ModuleInfoWithBalance(ModuleInfo):
    balance: int


class ModuleInfoWithOptionalBalance(ModuleInfo):
    balance: int | None



def bytes2str( data: bytes, mode: str = 'utf-8') -> str:
    
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)

def python2str( input):
    from copy import deepcopy
    import json
    input = deepcopy(input)
    input_type = type(input)
    if input_type == str:
        return input
    if input_type in [dict]:
        input = json.dumps(input)
    elif input_type in [bytes]:
        input = bytes2str(input)
    elif input_type in [list, tuple, set]:
        input = json.dumps(list(input))
    elif input_type in [int, float, bool]:
        input = str(input)
    return input

def dict2str(cls, data: str) -> str:
    import json
    return json.dumps(data)

def bytes2dict(data: bytes) -> str:
    import json
    data = bytes2str(data)
    return json.loads(data)

def str2bytes( data: str, mode: str = 'hex') -> bytes:
    if mode in ['utf-8']:
        return bytes(data, mode)
    elif mode in ['hex']:
        return bytes.fromhex(data)

def bytes2str( data: bytes, mode: str = 'utf-8') -> str:
    
    if hasattr(data, 'hex'):
        return data.hex()
    else:
        if isinstance(data, str):
            return data
        return bytes.decode(data, mode)

def str2python(input)-> dict:
    assert isinstance(input, str), 'input must be a string, got {}'.format(input)
    try:
        output_dict = json.loads(input)
    except json.JSONDecodeError as e:
        return input

    return output_dict


def detailed_error(e) -> dict:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name,
        'line_no': line_no,
        'line_text': line_text
    }   
    return response
    

    
    @classmethod
    def determine_type(cls, x):
        if x.lower() == 'null' or x == 'None':
            return None
        elif x.lower() in ['true', 'false']:
            return bool(x.lower() == 'true')
        elif x.startswith('[') and x.endswith(']'):
            # this is a list
            try:
                list_items = x[1:-1].split(',')
                # try to convert each item to its actual type
                x =  [cls.determine_type(item.strip()) for item in list_items]
                if len(x) == 1 and x[0] == '':
                    x = []
                return x
            except:
                # if conversion fails, return as string
                return x
        elif x.startswith('{') and x.endswith('}'):
            # this is a dictionary
            if len(x) == 2:
                return {}
            try:
                dict_items = x[1:-1].split(',')
                # try to convert each item to a key-value pair
                return {key.strip(): cls.determine_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
            except:
                # if conversion fails, return as string
                return x
        else:
            # try to convert to int or float, otherwise return as string
            try:
                return int(x)
            except ValueError:
                try:
                    return float(x)
                except ValueError:
                    return x