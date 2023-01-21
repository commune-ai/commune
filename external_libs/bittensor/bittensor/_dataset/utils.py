"""
Utilities for the DataImplementation
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import asyncio
import json
import os
import aiofiles




def sync_wrapper(fn:'asyncio.callable') -> 'callable':
    '''
    Convert Async funciton to Sync.

    Args:
        fn (callable): 
            An asyncio function.

    Returns: 
        wrapper_fn (callable):
            Synchronous version of asyncio function.
    '''
    def wrapper_fn(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return  wrapper_fn





async def async_load_json(path):

    path = ensure_json_path(path, ensure_directory=False, ensure_extension=True)
    data = json.loads(await async_read(path))
    return data
read_json = load_json = get_json = sync_wrapper(async_load_json)

async def async_save_json( path, data):
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
put_json = save_json = sync_wrapper(async_save_json)

async def async_read(path, mode='r'):
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data

async def async_write(path, data,  mode ='w'):
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)


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


async def async_read(path, mode='r'):
    async with aiofiles.open(path, mode=mode) as f:
        data = await f.read()
    return data
async def async_write(path, data,  mode ='w'):
    async with aiofiles.open(path, mode=mode) as f:
        await f.write(data)

def sync_wrapper(fn):
    def wrapper_fn(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return  wrapper_fn