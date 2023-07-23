""" An interface for serializing and deserializing bittensor tensors"""

# I DONT GIVE A FUCK LICENSE (IDGAF)
# Do whatever you want with this code
# Dont pull up with your homies if it dont work.
import numpy as np
import torch
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional
import sys
import os
import asyncio
from copy import deepcopy
from munch import Munch

from commune.modules.server.grpc.proto import DataBlock
import commune as c
import json


class Serializer(c.Module):
    r""" Bittensor base serialization object for converting between DataBlock and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """
    @staticmethod
    def resolve_key_list(x):
        if isinstance(x, dict):
            k_list = list(x.keys())
        elif isinstance(x, list):
            k_list = list(range(len(x)))
        elif type(x) in [tuple, set]: 
            # convert to list, to format as json
            x = list(x) 
            k_list = list(range(len(x)))
        else:
            raise Exception(f'{type(x)} not supported to get the keylist fam')

    python_types = [int, bool, float, tuple, dict, list, str, type(None)]
    def serialize(self,x:dict):
        x = c.copy(x)
        k_list = []
        if isinstance(x, dict):
            k_list = list(x.keys())
        elif isinstance(x, list):
            k_list = list(range(len(x)))
        elif type(x) in [tuple, set]: 
            # convert to list, to format as json
            x = list(x) 
            k_list = list(range(len(x)))

        for k in k_list:
            v = x[k]
            v_type = type(v)
            if v_type in [dict, list, tuple, set]:
                x[k] = self.serialize(x=v)
            else:
                str_v_type = self.get_str_type(data=v)
                if hasattr(self, f'serialize_{str_v_type}'):
                    v = getattr(self, f'serialize_{str_v_type}')(data=v)
                    x[k] = {'data': v, 'data_type': str_v_type,  'serialized': True}

        return x


    def is_serialized(self, data):
        if isinstance(data, dict) and \
                data.get('serialized', False) == True and \
                    'data' in data and 'data_type' in data:
            return True
        else:
            return False

    def deserialize(self, x) -> object:
        """Serializes a torch object to DataBlock wire format.
        """

        k_list = []
        if isinstance(x, dict):
            k_list = list(x.keys())
        elif isinstance(x, list):
            k_list = list(range(len(x)))
        elif type(x) in [tuple, set]: 
            # convert to list, to format as json
            x = list(x) 
            k_list = list(range(len(x)))

        for k in k_list:
            v = x[k]
            c.print(k, self.is_serialized(v))
            if type(v) in [dict, list, tuple, set]:
                x[k] = self.deserialize(x=v)
            if self.is_serialized(v):
                data_type = v['data_type']
                data = v['data']
                if hasattr(self, f'deserialize_{data_type}'):
                    x[k] = getattr(self, f'deserialize_{data_type}')(data=data)

        return x

    """
    ################ BIG DICT LAND ############################
    """
    
    def serialize_dict(self, data: dict) -> 'DataBlock':
        data = self.dict2bytes(data=data)
        return  data

    def deserialize_dict(self, data: bytes) -> 'DataBlock':
        data = self.bytes2dict(data=data)
        return data

    def serialize_bytes(self, data: dict) -> 'DataBlock':
        return  data
    
    def deserialize_bytes(self, data: bytes) -> 'DataBlock':
        return data

    def serialize_munch(self, data: dict) -> 'DataBlock':
        data=self.munch2dict(data)
        data = self.dict2bytes(data=data)
        return  data

    def deserialize_munch(self, data: bytes) -> 'DataBlock':
        data = self.bytes2dict(data=data)
        data = self.dict2munch(data)
        return data

    def dict2bytes(self, data:dict) -> bytes:
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes
    
    @classmethod
    def bytes2str(cls, x, **kwargs):
        return msgpack.unpackb(x, **kwargs)
    
    @classmethod
    def str2bytes(cls, x, **kwargs):
        return msgpack.packb(x, **kwargs)
        


    def bytes2dict(self, data:bytes) -> dict:
        json_object_bytes = msgpack.unpackb(data)
        return json.loads(json_object_bytes)

    """
    ################ BIG TORCH LAND ############################
    """
    def torch2bytes(self, data:torch.Tensor)-> bytes:
        return self.numpy2bytes(self.torch2numpy(data))
    
    def torch2numpy(self, data:torch.Tensor)-> np.ndarray:
        if data.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()
        return data


    def numpy2bytes(self, data:np.ndarray)-> bytes:
        output = msgpack.packb(data, default=msgpack_numpy.encode)
        return output
    
    def bytes2torch(self, data:bytes, ) -> torch.Tensor:
        numpy_object = self.bytes2numpy(data)
        
        int64_workaround = bool(numpy_object.dtype == np.int64)
        if int64_workaround:
            numpy_object = numpy_object.astype(np.float64)
        torch_object = torch.tensor(numpy_object)
        if int64_workaround:
            dtype = torch.int64
        return torch_object
    
    def bytes2numpy(self, data:bytes) -> np.ndarray:
        output = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
        return output

    def serialize_torch(self, data: torch.Tensor) -> DataBlock:

        output =   self.torch2bytes(data=data)
        return output
    
    def deserialize_torch(self, data: bytes) -> torch.Tensor:
        data =  self.bytes2torch(data=data )
        return data

    def get_str_type(self, data):
        data_type = str(type(data)).split("'")[1]
        if data_type in ['munch.Munch', 'Munch']:
            data_type = 'munch'
        if data_type in ['torch.Tensor', 'Tensor']:
            data_type = 'torch'
        
        return data_type

    @classmethod
    def test_serialize(cls):
        module = Serializer()
        data = {'bro': {'fam': torch.ones(2,2), 'bro': [torch.ones(1,1)]}}
        proto = module.serialize(data)
        module.deserialize(proto)

    @classmethod
    def test_deserialize(cls):
        module = Serializer()
        
        t = c.time()
        data = {'bro': {'fam':[[torch.randn(100,1000), torch.randn(100,1000)]], 'bro': [torch.ones(1,1)]}}
        proto = module.serialize(data)
        data = module.deserialize(proto)
        c.print(t - c.time())
        
        # return True
    
    @classmethod
    def test(cls):
        module = cls()
        stats = {}
        data = {'bro': {'fam': torch.randn(1000,2000), 'bro': [torch.ones(1,1)]}}
        # c.print(type(c.dict_get(data, 'bro.fam.data')))
        

        t = c.time()
        serialized_data = module.serialize(data)
        # c.print('proto', proto)
        deserialized_data = module.deserialize(serialized_data)
    
        assert deserialized_data['bro']['fam'].shape == data['bro']['fam'].shape
        assert deserialized_data['bro']['bro'][0].shape == data['bro']['bro'][0].shape

        stats['elapsed_time'] = c.time() - t
        stats['size_bytes'] = c.sizeof(data)
        stats['size_bytes_compressed'] = c.sizeof(serialized_data)
        stats['compression_ratio'] = stats['size_bytes'] / stats['size_bytes_compressed']
        stats['mb_per_second'] = c.round((stats['size_bytes'] / stats['elapsed_time']) / 1e6, 3)
        c.print(stats)
        
