# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import commune
from commune.subspace import Balance
import torch
from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.utils.ss58 import ss58_encode
from enum import Enum


custom_rpc_type_registry = {
    "types": {
        "SubnetInfo": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "Compact<u16>"],
                ["immunity_period", "Compact<u16>"],
                ["max_allowed_validators", "Compact<u16>"],
                ["min_allowed_weights", "Compact<u16>"],
                ["max_weights_limit", "Compact<u16>"],
                ["subnetwork_n", "Compact<u16>"],
                ["max_allowed_uids", "Compact<u16>"],
                ["blocks_since_last_step", "Compact<u64>"],
                ["tempo", "Compact<u16>"],
                ["network_connect", "Vec<[u16; 2]>"],
                ["emission_values", "Compact<u64>"],
            ]
        },
        "NeuronInfo": {
            "type": "struct",
            "type_mapping": [
                ["key", "AccountId"],
                ["uid", "Compact<u16>"],
                ["netuid", "Compact<u16>"],
                ["active", "bool"],
                ["stake", "Vec<(AccountId, Compact<u64>)>"],
                ["rank", "Compact<u16>"],
                ["emission", "Compact<u64>"],
                ["incentive", "Compact<u16>"],
                ["dividends", "Compact<u16>"],
                ["last_update", "Compact<u64>"],
                ["weights", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["bonds", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["pruning_score", "Compact<u16>"]
            ],
        },
        "AxonInfo": {
            "type": "struct",
            "type_mapping": [
                ["block", "u64"],
                ["ip", "u128"],
                ["port", "u16"],
            ],
        },
    }   
}

class ChainDataType(Enum):
    NeuronInfo = 1
    SubnetInfo = 2
# Constants
NANOPERTOKEN = 1e9
U8_MAX = 255
U16_MAX = 65535
U32_MAX = 4294967295
U64_MAX = 18446744073709551615
U128_MAX = 340282366920938463463374607431768211455
def from_scale_encoding( vec_u8: List[int], type_name: ChainDataType, is_vec: bool = False, is_option: bool = False ) -> Optional[Dict]:
    as_bytes = bytes(vec_u8)
    as_scale_bytes = ScaleBytes(as_bytes)
    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

    type_string = type_name.name

    if is_option:
        type_string = f'Option<{type_string}>'
    if is_vec:
        type_string = f'Vec<{type_string}>'

    obj = rpc_runtime_config.create_scale_object(
        type_string,
        data=as_scale_bytes
    )

    return obj.decode()

# Dataclasses for chain data.
@dataclass
class NeuronInfo:
    r"""
    Dataclass for neuron metadata.
    """
    key: str
    uid: int
    netuid: int
    active: int    
    # mapping of coldkey to amount staked to this Neuron
    stake: Dict[str, Balance]
    rank: float
    emission: float
    incentive: float
    dividends: float
    last_update: int
    weights: List[List[int]]
    bonds: List[List[int]]
    # axon_info: 'AxonInfo'
    pruning_score : int = 0

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> 'NeuronInfo':
        r""" Fixes the values of the NeuronInfo object.
        """
        neuron_info_decoded['key'] = ss58_encode(neuron_info_decoded['key'], commune.subspace.__ss58_format__)
        neuron_info_decoded['stake'] = { ss58_encode( key, commune.__ss58_format__): commune.subspace.Balance.from_nano(int(stake)) for key, stake in neuron_info_decoded['stake'] }
        neuron_info_decoded['weights'] = [[int(weight[0]), int(weight[1])] for weight in neuron_info_decoded['weights']]
        neuron_info_decoded['bonds'] = [[int(bond[0]), int(bond[1])] for bond in neuron_info_decoded['bonds']]
        neuron_info_decoded['rank'] = commune.subspace.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['rank'])
        neuron_info_decoded['emission'] = neuron_info_decoded['emission'] / NANOPERTOKEN
        neuron_info_decoded['incentive'] = commune.subspace.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['incentive'])
        neuron_info_decoded['dividends'] = commune.subspace.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['dividends'])
        neuron_info_decoded['pruning_score'] = commune.subspace.utils.U16_NORMALIZED_FLOAT(neuron_info_decoded['pruning_score'])

        # neuron_info_decoded['axon_info'] = AxonInfo.fix_decoded_values(neuron_info_decoded['axon_info'])

        return cls(**neuron_info_decoded)
    
    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> 'NeuronInfo':
        r""" Returns a NeuronInfo object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return NeuronInfo._null_neuron()
        
        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfo)
        if decoded is None:
            return NeuronInfo._null_neuron()
        
        decoded = NeuronInfo.fix_decoded_values(decoded)

        return decoded
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['NeuronInfo']:
        r""" Returns a list of NeuronInfo objects from a vec_u8.
        """
        
        decoded_list = from_scale_encoding(vec_u8, ChainDataType.NeuronInfo, is_vec=True)
        if decoded_list is None:
            return []

        decoded_list = [NeuronInfo.fix_decoded_values(decoded) for decoded in decoded_list]
        return decoded_list


    @staticmethod
    def _null_neuron() -> 'NeuronInfo':
        neuron = NeuronInfo(
            uid = 0,
            netuid = 0,
            active =  0,
            stake = {},
            # total_stake = Balance.from_nano(0),
            rank = 0,
            emission = 0,
            incentive = 0,
            dividends = 0,
            last_update = 0,
            weights = [],
            bonds = [],
            # axon_info = None,
            # is_null = True,
            key = "000000000000000000000000000000000000000000000000",
            pruning_score = 0,
        )
        return neuron

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> 'NeuronInfo':
        neuron = NeuronInfo( **neuron_dict )
        neuron.stake = { k: Balance.from_nano(stake) for k, stake in neuron.stake.items() }
        # neuron.total_stake = Balance.from_nano(neuron.total_stake)
        neuron.rank = neuron.rank / U16_MAX
        neuron.incentive = neuron.incentive / U16_MAX
        neuron.dividends = neuron.dividends / U16_MAX
        neuron.emission = neuron.emission / NANOPERTOKEN   
        return neuron
        
@dataclass
class AxonInfo:
    r"""
    Dataclass for axon info.
    """
    block: int
    ip: str
    port: int

    @classmethod
    def fix_decoded_values(cls, axon_info_decoded: Dict) -> 'AxonInfo':
        r""" Returns an AxonInfo object from an axon_info_decoded dictionary.
        """
        axon_info_decoded['ip'] = commune.utils.networking.int_to_ip(int(axon_info_decoded['ip']))
                                                                       
        return cls(**axon_info_decoded)

@dataclass
class SubnetInfo:
    r"""
    Dataclass for subnet info.
    """
    netuid: int
    immunity_period: int
    validator_epochs_per_reset: int
    validator_epoch_length: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weight_limit: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    tempo: int
    # netuid -> topk percentile prunning score requirement (u16:MAX normalized.)
    connection_requirements: Dict[str, float]
    emission_value: float

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional['SubnetInfo']:
        r""" Returns a SubnetInfo object from a vec_u8.
        """
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo)

        if decoded is None:
            return None
        
        return SubnetInfo.fix_decoded_values(decoded)
    
    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List['SubnetInfo']:
        r""" Returns a list of SubnetInfo objects from a vec_u8.
        """
        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo, is_vec=True, is_option=True)

        if decoded is None:
            return []
        
        decoded = [SubnetInfo.fix_decoded_values(d) for d in decoded]

        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> 'SubnetInfo':
        r""" Returns a SubnetInfo object from a decoded SubnetInfo dictionary.
        """
        return SubnetInfo(
            netuid = decoded['netuid'],
            immunity_period = decoded['immunity_period'],
            validator_epochs_per_reset = decoded['validator_epochs_per_reset'],
            validator_epoch_length = decoded['validator_epoch_length'],
            max_allowed_validators = decoded['max_allowed_validators'],
            min_allowed_weights = decoded['min_allowed_weights'],
            max_weight_limit = decoded['max_weights_limit'],
            subnetwork_n = decoded['subnetwork_n'],
            max_n = decoded['max_allowed_uids'],
            blocks_since_epoch = decoded['blocks_since_last_step'],
            tempo = decoded['tempo'],
            connection_requirements = {
                str(int(netuid)): commune.subspace.utils.U16_NORMALIZED_FLOAT(int(req)) for netuid, req in decoded['network_connect']
            },
            emission_value= decoded['emission_values'],
        )
    
    def to_parameter_dict( self ) -> 'torch.nn.ParameterDict':
        r""" Returns a torch tensor of the subnet info.
        """
        return torch.nn.ParameterDict( 
            self.__dict__
        )
    
    @classmethod
    def from_parameter_dict( cls, parameter_dict: 'torch.nn.ParameterDict' ) -> 'SubnetInfo':
        r""" Returns a SubnetInfo object from a torch parameter_dict.
        """
        return cls( **dict(parameter_dict) )
