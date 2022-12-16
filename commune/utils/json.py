
from .asyncio import async_read, async_write, sync_wrapper
import asyncio
import json
import os

async def async_get_json(path, return_type='dict'):
    try:  
        
        data = json.loads(await async_read(path))
    except FileNotFoundError as e:
        if handle_error:
            return None
        else:
            raise e

    if return_type in ['dict', 'json']:
        data = data
    elif return_type in ['pandas', 'pd']:
        data = pd.DataFrame(data)
    elif return_type in ['torch']:
        torch.tensor
    return data

read_json = load_json = get_json = sync_wrapper(async_get_json)

async def async_put_json( path, data):
        # Directly from dictionary
    path = ensure_path(path)
    data_type = type(data)
    if data_type in [dict, list, tuple, set, float, str, int]:
        json_str = json.dumps(data)
    elif data_type in [pd.DataFrame]:
        json_str = json.dumps(data.to_dict())
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    
    return await async_write(path, json_str)

put_json = save_json = sync_wrapper(async_put_json)

def path_exists(path:str):
    return os.path.exists(path)

def ensure_path( path):
    """
    ensures a dir_path exists, otherwise, it will create it 
    """

    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    return path
