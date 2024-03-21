""" An interface for serializing and deserializing bittensor tensors"""

# I DONT GIVE A FUCK LICENSE (IDGAF)
# Do whatever you want with this code
# Dont pull up with your homies if it dont work.
import numpy as np
import torch
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional
from copy import deepcopy
from munch import Munch

import commune as c
import json


class Serializer(c.Module):


    
    def serialize(self,x:dict, mode = 'str'):
        x = c.copy(x)
        x_type = type(x)
        if x_type in [dict, list, set, tuple]:
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
                x[k] = self.resolve_value(v=x[k])
        else:
            x = self.resolve_value(v=x)

        if mode == 'str':
            if isinstance(x, dict):
                x = self.dict2str(x)
        elif mode == 'bytes':
            if isinstance(x, dict):
                x = self.dict2bytes(x)
            elif isinstance(x, str):
                x = self.str2bytes(x)
        elif mode == 'dict' or mode == None:
            x = x
        else:
            raise Exception(f'{mode} not supported') 
        return x

    def resolve_value(self, v):
        new_value = None
        v_type = type(v)
        if v_type in [dict, list, tuple, set]:
            new_value = self.serialize(x=v, mode=None)
        else:
            # GET THE TYPE OF THE VALUE
            str_v_type = self.get_type_str(data=v)
            if hasattr(self, f'serialize_{str_v_type}'):
                # SERIALIZE MODE ON
                v = getattr(self, f'serialize_{str_v_type}')(data=v)
                new_value = {'data': v, 'data_type': str_v_type,  'serialized': True}
            else:
                # SERIALIZE MODE OFF
                new_value = v

        return new_value
        

    def is_serialized(self, data):
        if isinstance(data, dict) and data.get('serialized', False) and \
                    'data' in data and 'data_type' in data:
            return True
        else:
            return False

    def deserialize(self, x) -> object:
        """Serializes a torch object to DataBlock wire format.
        """
        if isinstance(x, dict) and isinstance(x.get('data', None), str):
            x = x['data']
        if isinstance(x, str):
            if x.startswith('{') or x.startswith('['):
                x = self.str2dict(x)
            else:
                return x
        
        is_single = isinstance(x,dict) and all([k in x for k in ['data', 'data_type', 'serialized']])
        if is_single:
            x = [x]
        k_list = []
        if isinstance(x, dict):
            k_list = list(x.keys())
        elif type(x) in [list]:
            k_list = list(range(len(x)))
        elif type(x) in [tuple, set]: 
            # convert to list, to format as json
            x = list(x) 
            k_list = list(range(len(x)))

        for k in k_list:
            v = x[k]
            if self.is_serialized(v):
                data_type = v['data_type']
                data = v['data']
                if hasattr(self, f'deserialize_{data_type}'):
                    x[k] = getattr(self, f'deserialize_{data_type}')(data=data)
            elif type(v) in [dict, list, tuple, set]:
                x[k] = self.deserialize(x=v)
        if is_single:
            x = x[0]
        return x

    """
    ################ BIG DICT LAND ############################
    """

    def serialize_pandas(self, data: 'pd.DataFrame') -> 'DataBlock':
        data = data.to_dict()
        data = self.dict2bytes(data=data)
        return data
    
    def deserialize_pandas(self, data: bytes) -> 'pd.DataFrame':
        data = self.bytes2dict(data=data)
        data = pd.DataFrame.from_dict(data)
        return data
    
    def serialize_dict(self, data: dict) -> str :
        data = self.dict2bytes(data=data)
        return  data

    def deserialize_dict(self, data: bytes) -> dict:
        data = self.bytes2dict(data=data)
        return data

    def serialize_bytes(self, data: dict) -> bytes:
        return self.bytes2str(data)
        
    def deserialize_bytes(self, data: bytes) -> 'DataBlock':
        if isinstance(data, str):
            data = self.str2bytes(data)
        return data

    def serialize_munch(self, data: dict) -> str:
        data=self.munch2dict(data)
        data = self.dict2str(data=data)
        return  data

    def deserialize_munch(self, data: bytes) -> 'Munch':
        return self.dict2munch(self.str2dict(data))

    def dict2bytes(self, data:dict) -> bytes:
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes
    def dict2str(self, data:dict) -> bytes:
        data_json_str = json.dumps(data)
        return data_json_str
    def str2dict(self, data:str) -> bytes:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return json.loads(data)
    
    @classmethod
    def hex2str(cls, x, **kwargs):
        return x.hex()

    bytes2str = hex2str
    
    @classmethod
    def str2hex(cls, x, **kwargs):
        return bytes.fromhex(x)

    str2bytes = str2hex
        


    def bytes2dict(self, data:bytes) -> dict:
        json_object_bytes = msgpack.unpackb(data)
        return json.loads(json_object_bytes)

    """
    ################ BIG TORCH LAND ############################
    """
    def torch2bytes(self, data:'torch.Tensor')-> bytes:
        return self.numpy2bytes(self.torch2numpy(data))
    
    def torch2numpy(self, data:'torch.Tensor')-> np.ndarray:
        if data.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()
        return data


    def numpy2bytes(self, data:np.ndarray)-> bytes:
        output = msgpack.packb(data, default=msgpack_numpy.encode)
        return output
    
    def bytes2torch(self, data:bytes, ) -> 'torch.Tensor':
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

    
    def deserialize_torch(self, data: dict) -> torch.Tensor:
        from safetensors.torch import load
        if isinstance(data, str):
            data = self.str2bytes(data)
        data = load(data)
        return data['data']



    def serialize_torch(self, data: torch.Tensor) -> 'DataBlock':     
        from safetensors.torch import save
        output = save({'data':data})  
        return self.bytes2str(output)


    def serialize_numpy(self, data: 'np.ndarray') -> 'np.ndarray':     
        data =  self.numpy2bytes(data)
        return self.bytes2str(data)

    def deserialize_numpy(self, data: bytes) -> 'np.ndarray':     
        if isinstance(data, str):
            data = self.str2bytes(data)
        return self.bytes2numpy(data)



    def get_type_str(self, data):
        '''
        ## Documentation for get_type_str function
        
        ### Purpose
        The purpose of this function is to determine and return the data type of the input given to it in string format. It supports identification of various data types including Munch, Tensor, ndarray, and DataFrame.
        
        ### Parameters
        - `self`: The instance of the class calling this function.
        - `data`: The input data whose type needs to be identified.
        
        ### Returns
        - `data_type`: A string representing the type of `data`. It can be one of the following:
          - 'munch' if the input data is of type Munch
          - 'torch' if the input data is a Torch tensor
          - 'numpy' if the input data is a NumPy ndarray
          - 'pandas' if the input data is a Pandas DataFrame
          - The actual type of the data as a string if it does not match any of the above
        
        ### Example Usage
        ```python
        my_class_instance = MyClass()
        data_type = my_class_instance.get_type_str(my_data)
        print(f"The data type is {data_type}")
        ```
        
        ### Notes
        This function utilizes Python's `type()` built-in function and string manipulation to parse and determine the data type. It simplifies type checking for specific common data types used in data science and machine learning applications.
        '''
        data_type = str(type(data)).split("'")[1]
        if 'Munch' in data_type:
            data_type = 'munch'
        if 'Tensor' in data_type or 'torch' in data_type:
            data_type = 'torch'
        if 'ndarray' in data_type:
            data_type = 'numpy'
        if  'DataFrame' in data_type:
            data_type = 'pandas'
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
    def test(cls, size=1):
        self = cls()
        stats = {}
        data = {'bro': {'fam': torch.randn(size,size), 'bro': [torch.ones(1000,500)] , 'bro2': [np.ones((100,1000))]}}

        t = c.time()
        serialized_data = self.serialize(data)
        assert isinstance(serialized_data, str), f"serialized_data must be a str, not {type(serialized_data)}"
        deserialized_data = self.deserialize(serialized_data)
    
        assert deserialized_data['bro']['fam'].shape == data['bro']['fam'].shape
        assert deserialized_data['bro']['bro'][0].shape == data['bro']['bro'][0].shape

        stats['elapsed_time'] = c.time() - t
        stats['size_bytes'] = c.sizeof(data)
        stats['size_bytes_compressed'] = c.sizeof(serialized_data)
        stats['size_deserialized_data'] = c.sizeof(deserialized_data)
        stats['compression_ratio'] = stats['size_bytes'] / stats['size_bytes_compressed']
        stats['mb_per_second'] = c.round((stats['size_bytes'] / stats['elapsed_time']) / 1e6, 3)

        data =  torch.randn(size,size)
        t = c.time()
        serialized_data = self.serialize(data)
        assert isinstance(serialized_data, str), f"serialized_data must be a str, not {type(serialized_data)}"
        deserialized_data = self.deserialize(serialized_data)
        assert deserialized_data.shape == data.shape

        stats['elapsed_time'] = c.time() - t
        stats['size_bytes'] = c.sizeof(data)
        stats['size_bytes_compressed'] = c.sizeof(serialized_data)
        stats['size_deserialized_data'] = c.sizeof(deserialized_data)
        stats['compression_ratio'] = stats['size_bytes'] / stats['size_bytes_compressed']
        stats['mb_per_second'] = c.round((stats['size_bytes'] / stats['elapsed_time']) / 1e6, 3)
        
        data =  np.random.randn(size,size)
        t = c.time()
        serialized_data = self.serialize(data, mode='str')
        assert isinstance(serialized_data, str), f"serialized_data must be a str, not {type(serialized_data)}"
        deserialized_data = self.deserialize(serialized_data)
        assert deserialized_data.shape == data.shape

        stats['elapsed_time'] = c.time() - t
        stats['size_bytes'] = c.sizeof(data)
        stats['size_bytes_compressed'] = c.sizeof(serialized_data)
        stats['size_deserialized_data'] = c.sizeof(deserialized_data)
        stats['compression_ratio'] = stats['size_bytes'] / stats['size_bytes_compressed']
        stats['mb_per_second'] = c.round((stats['size_bytes'] / stats['elapsed_time']) / 1e6, 3)

        return stats