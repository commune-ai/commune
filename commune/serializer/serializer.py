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
        import safetensors
        if isinstance(data, str):
            data = self.str2bytes(data)
        data = safetensors.torch.load(data)
        return data['data']

    def serialize(self, data: 'torch.Tensor') -> 'DataBlock':     
        import safetensors
        return safetensors.torch.save({'data':data}).hex()

# get all the objects that end in Serializer

# get all the objects that end in Serializer
name2serializer = {k.split('Serializer')[0].lower():v for k,v in locals().items() if k.endswith('Serializer')}
# name2serializer



class Serializer(c.Module):

    name2serializer = name2serializer
    list_types = [list, set, tuple] # shit that you can turn into lists for json
    iterable_types = [list, set, tuple, dict] # 
    json_serializable_types = [int, float, str, bool, type(None)]


    def serialize(self,x:dict, mode = 'dict', copy_value = True):
        if copy_value:
            x = c.copy(x)
            
        if type(x) in self.iterable_types:
            k_list = []
            if isinstance(x, dict):
                k_list = list(x.keys())
            else:
                assert type(x) in self.list_types, f'{type(x)} not supported'
                k_list = list(range(len(x)))
                x = list(x) 
            for k in k_list:
                x[k] = self.serialize(x[k],mode=None)
            return x        
        v_type = type(x)

        if v_type in self.json_serializable_types:

            result = x

        else:
            # GET THE TYPE OF THE VALUE
            data_type = str(type(x)).split("'")[1]
            if 'Munch' in data_type:
                data_type = 'munch'
            if 'Tensor' in data_type or 'torch' in data_type:
                data_type = 'torch'
            if 'ndarray' in data_type:
                data_type = 'numpy'
            if  'DataFrame' in data_type:
                data_type = 'pandas'
            serializer = self.name2serializer[data_type]
            if not hasattr(serializer, 'date_type'):
                serializer = serializer()
                setattr(serializer, 'date_type', data_type)
                self.name2serializer[data_type] = serializer
            if serializer is not None:
                # SERIALIZE MODE ON
                result = {'data':  serializer.serialize(x), 
                             'data_type': serializer.date_type,  
                             'serialized': True}
            else:
                result = {"success": False, "error": f"Type {serializer.data_type} not supported"}

        result = self.resolve_serialized_result(result, mode=mode)
        return result


    def resolve_serialized_result(self, result, mode = 'str'):
        if mode == 'str':
            if isinstance(result, dict):
                result = json.dumps(result)
        elif mode == 'bytes':
            if isinstance(result, dict):
                result = self.dict2bytes(result)    
            elif isinstance(result, str):
                result = self.str2bytes(result)
        elif mode in ['dict' , 'nothing', None]:
            pass
        else:
            raise Exception(f'{mode} not supported')
        return result
    


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
                if c.is_int(x):
                    x = int(x)
                elif c.is_float(x):
                    x = float(x)
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

    def dict2bytes(self, data:dict) -> bytes:
        import msgpack
        data_json_str = json.dumps(data)
        data_json_bytes = msgpack.packb(data_json_str)
        return data_json_bytes
    

    
    def str2dict(self, data:str) -> bytes:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        if isinstance(data, str):
            data = json.loads(data)
        return data
