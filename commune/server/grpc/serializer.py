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

from commune..server.grpc.proto import DataBlock
from commune.utils.dict import dict_put, dict_get
import commune
import json

if os.getenv('USE_STREAMLIT'):
    import streamlit as stw

class Serializer(commune.Module):
    r""" Bittensor base serialization object for converting between DataBlock and their
    various python tensor equivalents. i.e. torch.Tensor or tensorflow.Tensor
    """

    def serialize (self, data: object, metadata:dict) -> DataBlock:
        data_type = self.get_str_type(data)
        sub_blocks = []

        data_type = data_type.replace('.', '_')
        
                
        if data_type in ['dict']:
            object_map = self.get_non_json_objects(x=data, object_map={})
            
            for k_index ,k in enumerate(object_map.keys()):
                v = object_map[k]
                block_ref_path = list(map(lambda x: int(x) if x.isdigit() else str(x), k.split('.')))
                k_metadata = {'block_ref_path': block_ref_path, 'block_ref_idx': k_index}
                sub_blocks.append(self.serialize(data=v, metadata=deepcopy(k_metadata)))
                dict_put(data, block_ref_path , k_metadata)

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

        output_dict = dict(data= data, metadata = metadata)
        return output_dict

    """
    ################ BIG DICT LAND ############################
    """

    def serialize_dict(self, data: dict, metadata:dict) -> DataBlock:
        data = self.dict2bytes(data=data)
        return  data,  metadata


    def deserialize_dict(self, data: bytes, metadata:dict) -> DataBlock:
        data = self.bytes2dict(data=data)
        return data

    def serialize_bytes(self, data: dict, metadata:dict) -> DataBlock:
        return  data,  metadata
    
    def deserialize_bytes(self, data: bytes, metadata:dict) -> DataBlock:
        return data

    def serialize_munch(self, data: dict, metadata:dict) -> DataBlock:
        data=self.munch2dict(data)
        data = self.dict2bytes(data=data)
        return  data,  metadata

    def deserialize_munch(self, data: bytes, metadata:dict) -> DataBlock:
        data = self.bytes2dict(data=data)
        data = self.dict2munch(data)
        return data


    def dict2bytes(self, data:dict) -> bytes:
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes

    def bytes2dict(self, data:bytes) -> dict:
        json_object_bytes = msgpack.unpackb(data)
        return json.loads(json_object_bytes)



    """
    ################ BIG TORCH LAND ############################
    """
    def torch2bytes(self, data:torch.Tensor)-> bytes:
        if data.requires_grad:
            data = data.detach()
        torch_numpy = np.array(data.cpu().tolist())
        # torch_numpy = data.cpu().numpy().copy()
        data_buffer = msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        return data_buffer


    def bytes2torch(self, data:bytes, shape:list, dtype:str, requires_grad:bool=False) -> torch.Tensor:
        numpy_object = msgpack.unpackb(data, object_hook=msgpack_numpy.decode).copy()

        int64_workaround = bool(numpy_object.dtype == np.int64)
        if int64_workaround:
            numpy_object = numpy_object.astype(np.float64)
        torch_object = torch.tensor(numpy_object).view(shape).requires_grad_(requires_grad)
        if int64_workaround:
            dtype = torch.int64
        torch_object =  torch_object.to(dtype)
        return torch_object


    def serialize_torch(self, data: torch.Tensor, metadata:dict) -> DataBlock:

        metadata['dtype'] = str(data.dtype)
        metadata['shape'] = list(data.shape)
        metadata['requires_grad'] = data.requires_grad
        data = self.torch2bytes(data=data)
        return  data,  metadata

    def serialize_torch_device(self, data: torch.Tensor, metadata:dict) -> DataBlock:
        metadata['dtype'] =  'torch.device'
        # convert torch device to an int
        data = data.index
        return  data,  metadata
    def deserrialize_torch_device(self, data: torch.Tensor, metadata:dict) -> DataBlock:
        
        return  torch.device(data),  metadata

    def deserialize_torch(self, data: bytes, metadata: dict) -> torch.Tensor:

        dtype = metadata['dtype']
        assert 'torch.' in dtype

        # if dtype == 'torch.int64':
        #     dtype = 'torch.float64'
        dtype = eval(dtype)
        shape = metadata['shape']
        requires_grad = metadata['requires_grad']
        data =  self.bytes2torch(data=data, shape=shape, dtype=dtype, requires_grad=requires_grad )


        return data

    def get_str_type(self, data):
        data_type = str(type(data)).split("'")[1]
        if data_type in ['munch.Munch', 'Munch']:
            data_type = 'munch'
        if data_type in ['torch.Tensor', 'Tensor']:
            data_type = 'torch'
        
        return data_type

    def get_non_json_objects(self,x:dict, object_map:dict=None, root_key:str=None, python_types:Optional[list]=[int, bool, float, tuple, dict, list, str, type(None)]):
        object_map = object_map if object_map != None else {}
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
            
            if v_type in [dict, list, tuple]:
                self.get_non_json_objects(x=v, object_map=object_map, root_key=current_root_key)



        return object_map
    
   
    def empty(self):
        """Returns an empty DataBlock message with the version"""
        return DataBlock()

    @classmethod
    def test_serialize(cls):
        module = Serializer()
        data = {'bro': {'fam': torch.ones(100,1000), 'bro': [torch.ones(1,1)]}}
        proto = module.serialize(data)

    @classmethod
    def test_deserialize(cls):
        module = Serializer()
        data = {'bro': {'fam':[[torch.ones(100,1000), torch.ones(100,1000)]], 'bro': [torch.ones(1,1)]}}
        proto = module.serialize(data)
        data = module.deserialize(proto)
        st.write(data)
        return True
    
    @classmethod
    def test(cls):
        for f in dir(cls):
            if f.startswith('test_'):
                getattr(cls, f)()
                
                
if __name__ == "__main__":
    Serializer.run()

