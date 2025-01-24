import commune as c
import json

class Serializer(c.Module):

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
                
        if type(x) in self.json_serializable_types:
            result = x
        else:
            data_type = self.get_data_type_string(x)
            serializer = self.get_serializer(data_type)
            result = {'data':  serializer.serialize(x), 
                            'data_type': serializer.date_type,  
                            'serialized': True}
        return self.process_output(result, mode=mode)


    def deserialize(self, x) -> object:
        """Serializes a torch object to DataBlock wire format.
        """
        if isinstance(x, str):
            if x.startswith('{') or x.startswith('['):
                x = self.str2dict(x)
            else:
                if c.is_int(x):
                    x = int(x)
                elif c.is_float(x):
                    x = float(x)
                return x
        is_serialized = self.is_serialized(x)
        if is_serialized:
            serializer = self.get_serializer(x['data_type'])
            return serializer.deserialize(x['data'])
        return x
    
    def get_data_type_string(self, x):
        # GET THE TYPE OF THE VALUE
        data_type = str(type(x)).split("'")[1].lower()
        if 'munch' in data_type:
            data_type = 'munch'
        if 'tensor' in data_type or 'torch' in data_type:
            data_type = 'torch'
        if 'ndarray' in data_type:
            data_type = 'numpy'
        if  'dataframe' in data_type:
            data_type = 'pandas'

        return data_type

    def str2bytes(self,  data: str, mode: str = 'hex') -> bytes:
        if mode in ['utf-8']:
            return bytes(data, mode)
        elif mode in ['hex']:
            return bytes.fromhex(data)


    def process_output(self, result, mode = 'str'):
        """
        process the output
        
        """
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

    def serializer_map(self):
        type_path = self.dirpath()
        module_paths = c.objects(type_path)
        return {p.split('.')[-2]: c.obj(p)() for p in module_paths}

    def types(self):
        return list(self.serializer_map().keys())
    

    def get_serializer(self, data_type):
        serializer_map = self.serializer_map()
        if data_type in serializer_map:
            serializer = serializer_map[data_type]
            if not hasattr(serializer, 'date_type'):
                setattr(serializer, 'date_type', data_type)
                serializer_map[data_type] = serializer
        else:
            raise TypeError(f'Type Not supported for serializeation ({data_type}) with ')
        return serializer

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
