import numbers
from typing import Callable, Union

import commune as c
import pandas
import requests
import torch
from substrateinterface import Keypair
from substrateinterface.utils import ss58
from typing import List, Dict, Union, Optional, Tuple

U16_MAX = 65535 # 2**16 - 1
U64_MAX = 18446744073709551615 # 2**64 - 1
NANOPERTOKEN = 1e9

def strtobool(val: str) -> bool:
    """
    Converts a string to a boolean value.

    truth-y values are 'y', 'yes', 't', 'true', 'on', and '1';
    false-y values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def ss58_address_to_bytes(ss58_address: str) -> bytes:

    """Converts a ss58 address to a bytes object."""
    import scalecodec
    account_id_hex: str = scalecodec.ss58_decode(ss58_address, c.__ss58_format__)
    return bytes.fromhex(account_id_hex)

def u8_key_to_ss58(u8_key: List[int]) -> str:
    
    r"""
    Converts a u8-encoded account key to an ss58 address.

    Args:
        u8_key (List[int]): The u8-encoded account key.
    """
    import scalecodec
    # First byte is length, then 32 bytes of key.
    return scalecodec.ss58_encode( bytes(u8_key).hex(), bittensor.__ss58_format__)

def U16_NORMALIZED_FLOAT( x: int ) -> float:
    return float( x ) / float( U16_MAX ) 

def U64_NORMALIZED_FLOAT( x: int ) -> float:
    return float( x ) / float( U64_MAX )


def indexed_values_to_dataframe ( 
        prefix: Union[str, int],
        index: Union[list, torch.LongTensor], 
        values: Union[list, torch.Tensor],
        filter_zeros: bool = False
    ) -> 'pandas.DataFrame':
    # Type checking.
    if not isinstance(prefix, str) and not isinstance(prefix, numbers.Number):
        raise ValueError('Passed prefix must have type str or Number')
    if isinstance(prefix, numbers.Number):
        prefix = str(prefix)
    if not isinstance(index, list) and not isinstance(index, torch.Tensor):
        raise ValueError('Passed uids must have type list or torch.Tensor')
    if not isinstance(values, list) and not isinstance(values, torch.Tensor):
        raise ValueError('Passed values must have type list or torch.Tensor')
    if not isinstance(index, list):
        index = index.tolist()
    if not isinstance(values, list):
        values = values.tolist()

    index = [ idx_i for idx_i in index if idx_i < len(values) and idx_i >= 0 ]
    dataframe = pandas.DataFrame(columns=[prefix], index = index )
    for idx_i in index:
        value_i = values[ idx_i ]
        if value_i > 0 or not filter_zeros:
            dataframe.loc[idx_i] = pandas.Series( { str(prefix): value_i } )
    return dataframe


def unbiased_topk( values, k, dim=0, sorted = True, largest = True):
    r""" Selects topk as in torch.topk but does not bias lower indices when values are equal.
        Args:
            values: (torch.Tensor)
                Values to index into.
            k: (int):
                Number to take.
            
        Return:
            topk: (torch.Tensor):
                topk k values.
            indices: (torch.LongTensor)
                indices of the topk values.
    """
    permutation = torch.randperm(values.shape[ dim ])
    permuted_values = values[ permutation ]
    topk, indices = torch.topk( permuted_values,  k, dim = dim, sorted=sorted, largest=largest )
    return topk, permutation[ indices ]



def valid_ss58_address( address: str ) -> bool:
    """
    Checks if the given address is a valid ss58 address.

    Args:
        address(str): The address to check.

    Returns:
        True if the address is a valid ss58 address for Bittensor, False otherwise.
    """
    try:
        return ss58.is_valid_ss58_address( address, valid_ss58_format=c.__ss58_format__ )
    except (IndexError):
        return False

def is_valid_ed25519_pubkey( public_key: Union[str, bytes] ) -> bool:
    """
    Checks if the given public_key is a valid ed25519 key.

    Args:
        public_key(Union[str, bytes]): The public_key to check.

    Returns:    
        True if the public_key is a valid ed25519 key, False otherwise.
    
    """
    try:
        if isinstance( public_key, str ):
            if len(public_key) != 64 and len(public_key) != 66:
                raise ValueError( "a public_key should be 64 or 66 characters" )
        elif isinstance( public_key, bytes ):
            if len(public_key) != 32:
                raise ValueError( "a public_key should be 32 bytes" )
        else:
            raise ValueError( "public_key must be a string or bytes" )

        keypair = Keypair(
            public_key=public_key,
            ss58_format=commune.__ss58_format__
        )

        ss58_addr = keypair.ss58_address
        return ss58_addr is not None

    except (ValueError, IndexError):
        return False


def strtobool_with_default( default: bool ) -> Callable[[str], bool]:
    """
    Creates a strtobool function with a default value.

    Args:
        default(bool): The default value to return if the string is empty.

    Returns:
        The strtobool function with the default value.
    """
    return lambda x: strtobool(x) if x != "" else default


def strtobool(val: str) -> bool:
    """
    Converts a string to a boolean value.

    truth-y values are 'y', 'yes', 't', 'true', 'on', and '1';
    false-y values are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))



def get_ss58_format( ss58_address: str ) -> int:
    """Returns the ss58 format of the given ss58 address."""
    return ss58.get_ss58_format( ss58_address )

def strtobool_with_default( default: bool ) -> Callable[[str], bool]:
    """
    Creates a strtobool function with a default value.

    Args:
        default(bool): The default value to return if the string is empty.

    Returns:
        The strtobool function with the default value.
    """
    return lambda x: strtobool(x) if x != "" else default

""" Conversion for weight between chain representation and torch tensor
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

from typing import Tuple, List
import torch



def normalize_max_weight(  x: torch.FloatTensor, limit:float = 0.1 ) -> 'torch.FloatTensor':
    r""" Normalizes the tensor x so that sum(x) = 1 and the max value is not greater than the limit.
        Args:
            x (:obj:`torch.FloatTensor`):
                Tensor to be max_value normalized.
            limit: float:
                Max value after normalization.     
        Returns:
            y (:obj:`torch.FloatTensor`):
                Normalized x tensor.
    """
    epsilon = 1e-7 #For numerical stability after normalization
    
    weights =  x.clone()
    values, _ = torch.sort(weights)

    if x.sum() == 0 or len(x)*limit <= 1:
        return torch.ones_like(x)/x.size(0)
    else:
        estimation = values/values.sum()
        
        if estimation.max() <= limit:
            return weights/weights.sum()

        # Find the cumlative sum and sorted tensor
        cumsum = torch.cumsum(estimation,0)

        # Determine the index of cutoff
        estimation_sum = torch.tensor([(len(values)-i-1)*estimation[i] for i in range(len(values))])
        n_values = (estimation/(estimation_sum+cumsum+epsilon)<limit).sum()

        # Determine the cutoff based on the index        
        cutoff_scale = (limit*cumsum[n_values-1]-epsilon)/(1-(limit*(len(estimation)-n_values)))
        cutoff= cutoff_scale*values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights/weights.sum()

        return y

def convert_weight_uids_and_vals_to_tensor( n: int, uids: List[int], weights: List[int] ) -> 'torch.FloatTensor':
    r""" Converts weights and uids from chain representation into a torch tensor (inverse operation from convert_weights_and_uids_for_emit)
        Args:
            n: int:
                number of neurons on network.
            uids (:obj:`List[int],`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`List[int],`):
                Tensor of weights.
        Returns:
            row_weights ( torch.FloatTensor ):
                Converted row weights.
    """
    row_weights = torch.zeros( [ n ], dtype=torch.float32 )
    for uid_j, wij in list(zip( uids, weights )):
        row_weights[ uid_j ] = float( wij ) / float(U16_MAX)
    return row_weights

def convert_bond_uids_and_vals_to_tensor( n: int, uids: List[int], bonds: List[int] ) -> 'torch.LongTensor':
    r""" Converts bond and uids from chain representation into a torch tensor.
        Args:
            n: int:
                number of neurons on network.
            uids (:obj:`List[int],`):
                Tensor of uids as destinations for passed bonds.
            bonds (:obj:`List[int],`):
                Tensor of bonds.
        Returns:
            row_bonds ( torch.FloatTensor ):
                Converted row bonds.
    """
    row_bonds = torch.zeros( [ n ], dtype=torch.int64 )
    for uid_j, bij in list(zip( uids, bonds )):
        row_bonds[ uid_j ] = int( bij ) 
    return row_bonds

def convert_weights_and_uids_for_emit( uids: torch.LongTensor, weights: torch.FloatTensor ) -> Tuple[List[int], List[int]]:
    r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
        Args:
            uids (:obj:`torch.LongTensor,`):
                Tensor of uids as destinations for passed weights.
            weights (:obj:`torch.FloatTensor,`):
                Tensor of weights.
        Returns:
            weight_uids (List[int]):
                Uids as a list.
            weight_vals (List[int]):
                Weights as a list.
    """
    # Checks.
    weights = weights.tolist()
    uids = uids.tolist()
    if min(weights) < 0:
        raise ValueError('Passed weight is negative cannot exist on chain {}'.format(weights))
    if min(uids) < 0:
        raise ValueError('Passed uid is negative cannot exist on chain {}'.format(uids))
    if len(uids) != len(weights):
        raise ValueError('Passed weights and uids must have the same length, got {} and {}'.format(len(uids), len(weights)))
    if sum(weights) == 0:
        return [],[] # Nothing to set on chain.
    else:
        weights = [ float(value) / sum(weights) for value in weights] # Initial normalization.

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = int(float(weight_i) * int(U16_MAX))  # convert to int representation.

        # Filter zeros
        if uint16_val != 0: # Filter zeros
            weight_vals.append( uint16_val )
            weight_uids.append( uid_i ) 

    return weight_uids, weight_vals