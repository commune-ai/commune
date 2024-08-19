""" An interface for serializing and deserializing tensors"""

# I DONT GIVE A FUCK LICENSE (IDGAF)
# Do whatever you want with this code
# Dont pull up with your homies if it dont work.
import numpy as np
from typing import *
from copy import deepcopy
import commune as c
import json

class MunchSerializer:

    def serialize(self, data: dict) -> str:
        return  json.dumps(self.munch2dict(data))

    def deserialize(self, data: bytes) -> 'Munch':
        return self.dict2munch(self.str2dict(data))

    
    def str2dict(self, data:str) -> bytes:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        if isinstance(data, str):
            data = json.loads(data)
        return data

    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> 'Munch':
        from munch import Munch
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = cls.dict2munch(v)
            x = Munch(x)
        return x 

    @classmethod
    def munch2dict(cls, x:'Munch', recursive:bool=True)-> dict:
        from munch import Munch
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = cls.munch2dict(v)
        return x 
        

    def dict2str(self, data:dict) -> bytes:
        return


class BytesSerializer:

    def serialize(self, data: dict) -> bytes:
        return data.hex()
        
    def deserialize(self, data: bytes) -> 'DataBlock':
        if isinstance(data, str):
            data = bytes.fromhex(data)
        return data

class NumpySerializer:
    
    def serialize(self, data: 'np.ndarray') -> 'np.ndarray':     
        return  self.numpy2bytes(data).hex()

    def deserialize(self, data: bytes) -> 'np.ndarray':     
        if isinstance(data, str):
            data = bytes.fromhex(data)
        return self.bytes2numpy(data)

    def bytes2numpy(self, data:bytes) -> np.ndarray:
        import msgpack_numpy
        import msgpack
        output = msgpack.unpackb(data, object_hook=msgpack_numpy.decode)
        return output
    
    def numpy2bytes(self, data:np.ndarray)-> bytes:
        import msgpack_numpy
        import msgpack
        output = msgpack.packb(data, default=msgpack_numpy.encode)
        return output
    
    @classmethod
    def bytes2str(cls, x, **kwargs):
        return x.hex()
    
    @classmethod
    def str2bytes(cls, x, **kwargs):
        return 

class PandasSerializer:

    def serialize(self, data: 'pd.DataFrame') -> 'DataBlock':
        data = data.to_json()
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return data
    
    def deserialize(self, data: bytes) -> 'pd.DataFrame':
        import pandas as pd
        data = pd.DataFrame.from_dict(json.loads(data))
        return data
    
class TorchSerializer:
    def deserialize(self, data: dict) -> 'torch.Tensor':
        from safetensors.torch import load
        if isinstance(data, str):
            data = self.str2bytes(data)
        data = load(data)
        return data['data']

    def serialize(self, data: 'torch.Tensor') -> 'DataBlock':     
        from safetensors.torch import save
        return save({'data':data}).hex()

serilizer_map = {k.split('Serializer')[0].lower():v for k,v in locals().items() if k.endswith('Serializer')}
