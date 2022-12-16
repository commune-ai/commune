
from .asyncio import async_read, async_write, sync_wrapper
import asyncio
import yaml
import os

async def async_get_yaml(path, return_type='dict'):
    try:  
        
        data = yaml.load(await async_read(path))
    except FileNotFoundError as e:
        if handle_error:
            return None
        else:
            raise e

    if return_type in ['dict', 'yaml']:
        data = data
    elif return_type in ['pandas', 'pd']:
        data = pd.DataFrame(data)
    elif return_type in ['torch']:
        torch.tensor
    return data

read_yaml = load_yaml = get_yaml = sync_wrapper(async_get_yaml)

async def async_put_yaml( path, data):
        # Directly from dictionary
    path = ensure_path(path)
    data_type = type(data)
    if data_type in [dict, list, tuple, set, float, str, int]:
        yaml_str = yaml.dump(data)
    elif data_type in [pd.DataFrame]:
        yaml_str = yaml.dump(data.to_dict())
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    
    return await async_write(path, yaml_str)

put_yaml = save_yaml = sync_wrapper(async_put_yaml)

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
