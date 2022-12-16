""" An interface for serializing and deserializing bittensor tensors"""

# I DONT GIVE A FUCK LICENSE (IDGAF)
# Do whatever you want with this code
# Dont pull up with your homies if it dont work.

import torch
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional
import sys
import os
import asyncio
from copy import deepcopy
asyncio.set_event_loop(asyncio.new_event_loop())
sys.path.append(os.getenv('PWD'))
from commune.proto import DataBlock
import commune
import json
import streamlit as st
from commune.utils import dict_put, dict_get

class SerializerModule:
    r""" Bittensor base serialization object for converting between DataBlock and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    @staticmethod
    def empty():
        """Returns an empty DataBlock message with the version"""
        return DataBlock()

    def serialize (self, data: object, metadata:dict={}) -> DataBlock:
        data_type = self.get_str_type(data)
        sub_blocks = []
        if data_type == 'torch.Tensor':
            data_type = 'torch'
        if data_type == 'dict':
            object_map = self.get_non_json_objects(x=data)

            for k_index ,k in enumerate(object_map.keys()):
                v = object_map[k]
                k_metadata = {'block_ref_path': k, 'block_ref_idx': k_index}
                sub_blocks.append(self.serialize(data=v, metadata=deepcopy(k_metadata)))
                dict_put(data, k, k_metadata)

            # st.write(sub_blocks)

        serializer = getattr(self, f'serialize_{data_type}')
        data_bytes, metadata = serializer( data = data, metadata=metadata )
        metadata['data_type'] =  data_type
        metadata_bytes = self.dict2bytes(metadata)
        return DataBlock(data=data_bytes, metadata = metadata_bytes, blocks=sub_blocks)

    def deserialize(self, proto: DataBlock) -> object:
        """Serializes a torch object to DataBlock wire format.
        """
        metadata = self.bytes2dict(proto.metadata)
        data_type = metadata['data_type']
        deserializer = getattr(self, f'deserialize_{data_type}')


        data = deserializer( data = proto.data , metadata= metadata)
        if len(proto.blocks) > 0:
            for proto_block in proto.blocks:
                block = self.deserialize(proto=proto_block)
                dict_put(data, block['metadata']['block_ref_path'], block['data'])

        st.write(len(proto.blocks))
        
        output_dict = dict(data= data, metadata = metadata)
        return output_dict

    """
    ################ BIG DICT LAND ############################
    """

    def serialize_dict(self, data: dict, metadata:dict={}) -> DataBlock:
        data = self.dict2bytes(data=data)
        return  data,  metadata

    def deserialize_dict(self, data: bytes, metadata:dict={}) -> DataBlock:
        data = self.bytes2dict(data=data)
        return data

    @staticmethod
    def dict2bytes(data:dict={}) -> bytes:
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes

    @staticmethod 
    def bytes2dict( data:bytes) -> dict:
        json_object_bytes = msgpack.unpackb(data)
        return json.loads(json_object_bytes)


    """
    ################ BIG TORCH LAND ############################
    """


    def serialize_torch(self, data: torch.Tensor, metadata:dict={}) -> DataBlock:

        metadata['dtype'] = str(data.dtype)
        metadata['shape'] = list(data.shape)
        metadata['requires_grad'] = data.requires_grad
        data = self.torch2bytes(data=data)

        return  data,  metadata

    def deserialize_torch(self, data: bytes, metadata: dict) -> torch.Tensor:

        dtype = metadata['dtype']
        assert 'torch.' in dtype
        dtype = eval(dtype)
        shape = metadata['shape']
        requires_grad = metadata['requires_grad']
        data =  self.bytes2torch(data=data, shape=shape, dtype=dtype, requires_grad=requires_grad )
        return data
    @staticmethod
    def torch2bytes(data:torch.Tensor)-> bytes:
        torch_numpy = data.cpu().detach().numpy().copy()
        data_buffer = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        return data_buffer

    @staticmethod
    def bytes2torch(data:bytes, shape:list, dtype:str, requires_grad:bool=False) -> torch.Tensor:
        numpy_object = msgpack.unpackb(data, object_hook=msgpack_numpy.decode).copy()
        torch_object = torch.as_tensor(numpy_object).view(shape).requires_grad_(requires_grad)
        torch_object =  torch_object.type(dtype)
        return torch_object

    @staticmethod
    def get_str_type(data):
        return str(type(data)).split("'")[1]

    @classmethod
    def get_non_json_objects(cls,x:dict, object_map:dict = {}, root_key:str='', python_types:Optional[list]=[int, bool, float, tuple, dict, list]):

        k_list = []
        if isinstance(x, dict):
            k_list = list(x.keys())
        elif isinstance(x, list):
            k_list = list(range(len(x)))

        for k in k_list:
            v = x[k]
            v_type = type(v)

            current_root_key = f'{root_key}.{k}' if root_key else k

            if v_type not in python_types:
                object_map[current_root_key] = v
            
            if v_type in [dict, list]:
                cls.get_non_json_objects(x=v, object_map=object_map, root_key=current_root_key)


        return object_map
    
    
    
if __name__ == "__main__":
    module = SerializerModule()
    # data = {'bro': [10, 10, 10]}
    data = {'bro': {'fam': torch.ones(100,1000), 'bro': torch.ones(1,1)}}


    # st.write(data)

    # st.write(dict_get(data, 'bro.bro'))
    
    # st.write(module.get_non_json_objects(x=data))
    proto = module.serialize(data)
    st.write(module.deserialize(proto))
