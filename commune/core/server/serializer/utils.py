def dict2bytes( data:dict) -> bytes:
    import msgpack
    data_json_str = json.dumps(data)
    data_json_bytes = msgpack.packb(data_json_str)
    return data_json_bytes

def str2dict( data:str) -> bytes:
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    if isinstance(data, str):
        data = json.loads(data)
    return data

def get_data_type_string( x):
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
    else:
        raise Exception(f'{mode} not supported')