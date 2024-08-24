from .serializers import serilizer_map
import commune as c
class Serializer(c.Module):

    serilizer_map = serilizer_map
    serializers = serilizer_map.values()
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
            data_type = str(type(x)).split("'")[1].lower()

            if 'munch' in data_type:
                data_type = 'munch'
            if 'tensor' in data_type or 'torch' in data_type:
                data_type = 'torch'
            if 'ndarray' in data_type:
                data_type = 'numpy'
            if  'dataframe' in data_type:
                data_type = 'pandas'

            serializer = serilizer_map[data_type]
            if not hasattr(serializer, 'date_type'):
                serializer = serializer()
                setattr(serializer, 'date_type', data_type)
                serilizer_map[data_type] = serializer
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

    def test(self):
        import torch, time
        data_list = [
            torch.ones(1000),
            torch.zeros(1000),
            torch.rand(1000), 
            [1,2,3,4,5],
            {'a':1, 'b':2, 'c':3},
            'hello world',
            1,
            1.0,
            True,
            False,
            None

        ]
        for data in data_list:
            t1 = time.time()
            data = self.serialize(data)
            data = self.deserialize(data)
            t2 = time.time()
            latency = t2 - t1
            emoji = '✅' if data == data else '❌'
            print('DATA', data, 'LATENCY', latency, emoji)
        return {'msg': 'PASSED test_serialize_deserialize'}
