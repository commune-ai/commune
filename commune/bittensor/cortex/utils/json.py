
from .asyncio import async_read, async_write, sync_wrapper
import asyncio
import json
import os

async def async_get_json(path):

    path = ensure_json_path(path, ensure_directory=False, ensure_extension=True)
    data = json.loads(await async_read(path))
    return data
async_load_json = async_get_json
read_json = load_json = get_json = sync_wrapper(async_get_json)

async def async_put_json( path, data):
        # Directly from dictionary
    path = ensure_json_path(path, ensure_directory=True, ensure_extension=True)
    data_type = type(data)
    if data_type in [dict, list, tuple, set, float, str, int]:
        json_str = json.dumps(data)
    elif data_type in [pd.DataFrame]:
        json_str = json.dumps(data.to_dict())
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    
    return await async_write(path, json_str)
async_save_json = async_put_json 
put_json = save_json = sync_wrapper(async_put_json)

def path_exists(path:str):
    return os.path.exists(path)


def ensure_json_path( path, ensure_directory:bool=True, ensure_extension:bool=True)-> str:
    """
    ensures a dir_path exists, otherwise, it will create it 
    """


    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if not bool(os.path.splitext(path)[-1]):
        path = '.'.join([path, 'json'])

    return path
