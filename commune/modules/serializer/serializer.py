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

from commune.module.server.proto import DataBlock
import commune as c
import json


class Serializer(c.Module):
    r""" Bittensor base serialization object for converting between DataBlock and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    def serialize(self, data: object, mode='bytes') -> 'DataBlock':
        data = c.copy(data)
        data_type = self.get_str_type(data)
        block_ref_paths = []
        if data_type in ['dict']:
            object_map = self.get_non_json_objects(x=data, object_map={})
            for k_index ,k in enumerate(object_map.keys()):
                v = object_map[k]
                block_ref_path = list(map(lambda x: int(x) if x.isdigit() else str(x), k.split('.')))
                
                
                c.dict_put(data, block_ref_path , self.serialize(data=v, mode='str'))
                block_ref_paths.append(block_ref_path)
        else:
            data = getattr(self, f'serialize_{data_type}')(data)
            data = self.bytes2str(data) if mode == 'str' else data
        return {
            'data': data,
            'block_ref_paths': block_ref_paths,
            'data_type': data_type
        }


    def deserialize(self, data: 'DataBlock') -> object:
        """Serializes a torch object to DataBlock wire format.
        """
        
        block_ref_paths = data['block_ref_paths'] 
        data = data['data']
        if len(block_ref_paths) > 0:
            for block_ref_path in block_ref_paths:
                block = c.dict_get(data, block_ref_path)
                data_type = block['data_type']
                block['data'] = self.str2bytes(block['data'])
                deserialized_data = getattr(self, f'deserialize_{data_type}')(data =block['data'])
                c.dict_put(data, block_ref_path, deserialized_data)

        return data

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

    python_types = [int, bool, float, tuple, dict, list, str, type(None)]
    def get_non_json_objects(self,x:dict, 
                             object_map:dict=None, 
                             root_key:str=None, 
                             python_types:Optional[list]=python_types):
        object_map = object_map if object_map != None else {}
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

            current_root_key = f'{root_key}.{k}' if root_key else k

            if v_type not in python_types:
                object_map[current_root_key] = v
            
            if v_type in [dict, list, tuple]:
                self.get_non_json_objects(x=v, object_map=object_map, root_key=current_root_key)



        return object_map
    

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
        data = {'bro': {'fam': torch.randn(50,1000), 'bro': [torch.ones(1,1)]}}
        # c.print(type(c.dict_get(data, 'bro.fam.data')))
        

        t = c.time()
        proto = module.serialize(data)
        # c.print('proto', proto)
        module.deserialize(proto)

        stats['elapsed_time'] = c.time() - t
        stats['size_bytes'] = c.sizeof(data)
        stats['size_bytes_compressed'] = c.sizeof(proto)
        stats['compression_ratio'] = stats['size_bytes'] / stats['size_bytes_compressed']
        stats['mb_per_second'] = c.round((stats['size_bytes'] / stats['elapsed_time']) / 1e6, 3)
        c.print(stats)
        
