import os
import time
from time import  strftime
import random
import yaml
import json
from copy import deepcopy
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Union, Any, Tuple, Callable, Optional
from importlib import import_module
import pickle
import math
from typing import Union
import datetime
import munch
from commune.utils.asyncio import sync_wrapper
from commune.utils.os import ensure_path, path_exists
import pandas as pd

def rm_json(path:str, ignore_error:bool=True) -> Union['NoneType', str]:
    import shutil, os
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    else:
        if ignore_error:
            return None
        raise Exception(path, 'is not a valid path')
    
    return path
    

chunk_list = lambda my_list, n: [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]


"""

Methods for Getting Abstractions
=--

"""

def get_module(path,prefix = 'commune'):
    '''
    gets the object
    {module_path}.{object_name}
    ie.
    {model.block.nn.rnn}.{LSTM}
    '''
    assert isinstance(prefix, str)

    if prefix != path[:len(prefix)]:
        path = '.'.join([prefix, path])

    module_path = '.'.join(path.split('.'))

    try:
        module = import_module(module_path)
    except (ModuleNotFoundError) as e:
        if handle_failure :
            return None
        else:
            raise e 

    return module




def import_object(key):
    module_path = '.'.join(key.split('.')[:-1])
    module = import_module(module_path)
    object_name = key.split('.')[-1]
    obj = getattr(module, object_name)
    return obj



def get_object(path,prefix = 'commune'):
    '''
    gets the object
    {module_path}.{object_name}
    ie.
    {model.block.nn.rnn}.{LSTM}
    '''
    assert isinstance(prefix, str)

    if prefix != path[:len(prefix)]:
        path = '.'.join([prefix, path])

    object_name = path.split('.')[-1]

    try:
        module_class = import_object(path)
    except Exception as e:
        old_path = deepcopy(path)
        path = '.'.join(path.split('.')[:-1] + ['module', path.split('.')[-1]])
        print(f'Trying {path} instead of {old_path}')
        module_class = import_object(path)

    return module_class

def try_fn_n_times(fn, kwargs, try_count_limit):
    '''
    try a function n times
    '''
    try_count = 0
    return_output = None
    while try_count < try_count_limit:
        try:
            return_output = fn(**kwargs)
            break
        except RuntimeError:
            try_count += 1
    return return_output


def list2str(input):
    assert isinstance(input, list)
    return '.'.join(list(map(str, input)))


def string_replace(cfg, old_str, new_str):

    '''

    :param cfg: dictionary (from yaml)
    :param old_str: old string
    :param new_str: new string replacing old string
    :return:
        cfg after old string is replaced with new string
    '''
    if type(cfg) == dict:
        for k,v in cfg.items():
            if type(v) == str:
                # replace string if list
                if old_str in v:
                    cfg[k] = v.replace(old_str, new_str)
            elif type(v) in [list, dict]:
                # recurse if list or dict
                cfg[k] = string_replace(cfg=v,
                                         old_str=old_str,
                                         new_str=new_str)
            else:
                # for all other types in yaml files
                cfg[k] = v
    elif type(cfg) == list:
        for k, v in enumerate(cfg):
            if type(v) == str:
                # replace string if list
                if old_str in v:
                    cfg[k] = v.replace(old_str, new_str)
            elif type(v) in [list, dict]:
                # recurse if list or dict
                cfg[k] = string_replace(cfg=v,
                                         old_str=old_str,
                                         new_str=new_str)
            else:
                # for all other types in yaml files
                cfg[k] = v

    return cfg


def chunk(sequence,
          chunk_size=None,
          append_remainder=False,
          distribute_remainder=True,
          num_chunks= None):
    # Chunks of 1000 documents at a time.

    if chunk_size is None:
        assert (type(num_chunks) == int)
        chunk_size = len(sequence) // num_chunks

    if chunk_size >= len(sequence):
        return [sequence]
    remainder_chunk_len = len(sequence) % chunk_size
    remainder_chunk = sequence[:remainder_chunk_len]
    sequence = sequence[remainder_chunk_len:]
    sequence_chunks = [sequence[j:j + chunk_size] for j in range(0, len(sequence), chunk_size)]

    if append_remainder:
        # append the remainder to the sequence
        sequence_chunks.append(remainder_chunk)
    else:
        if distribute_remainder:
            # distributes teh remainder round robin to each of the chunks
            for i, remainder_val in enumerate(remainder_chunk):
                chunk_idx = i % len(sequence_chunks)
                sequence_chunks[chunk_idx].append(remainder_val)

    return sequence_chunks


def has_fn(obj, fn_name):
    return callable(getattr(obj, fn_name, None))

def even_number_split(number=10, splits=2):
    starting_bin_value = number // splits
    split_bins = splits * [starting_bin_value]
    left_over_number = number % starting_bin_value
    for i in range(splits):
        split_bins[i] += 1
        left_over_number -= 1
        if left_over_number == 0:
            break

    return split_bins



def dict_fn(input, fn=lambda x: x.shape[0]):
    # applies fn to leaf nodes of dict

    input = fn(input)


    if isinstance(input, dict):
        keys = list(input.keys())
    elif isinstance(input, list):
        keys = list(range(len(input)))
    else:
        return input


    for k in keys:
        v = input[k]
        input[k] = dict_fn(input=v, fn=fn)
    return input

def dict_delete(input_dict,keys ):
    """
    insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    else:
        assert isinstance(keys, list)
    
    key = keys[0]
    if key in input_dict:    
        if len(keys) == 1:
            assert isinstance(input_dict,dict), f"{keys}, {input_dict}"
            del input_dict[key]

        elif len(keys) > 1:
            dict_delete(input_dict=input_dict[key],
                                keys=keys[1:])
    else:
        return None
 
dict_pop = dict_del = dict_delete

def dict_has(input_dict,keys):
    """
    insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    selected_items = [input_dict]
    for k in keys:
        current_item = selected_items[-1]
        if (isinstance(current_item,dict) and k in current_item):
            selected_items.append(current_item[k])
            
        else:
            return False
    return True
            
def dict_get(input_dict,keys, default_value=False):
    """
    get keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        if keys=='':
            return input_dict
        keys = keys.split('.')
    


    assert isinstance(keys, list)
    if len(keys) == 0:
        return input_dict
        
    assert isinstance(keys[0], str)

    key = keys[0]
    try:

        next_object_list = [input_dict[key]]
        for key in keys[1:]:
            next_object_list += [next_object_list[-1][key]]
        return next_object_list[-1]
    except Exception as e:
        return default_value

def dict_put(input_dict,keys: Union[str, list], value: Any ):
    """
    insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    elif not type(keys) in [list, tuple]:
        keys = str(keys)
    key = keys[0]
    if len(keys) == 1:
        if  isinstance(input_dict,dict):
            input_dict[key] = value
        elif isinstance(input_dict, list):
            input_dict[int(key)] = value
        elif isinstance(input_dict, tuple):
            input_dict[int(key)] = value
            

    elif len(keys) > 1:
        if key not in input_dict:
            input_dict[key] = {}
        dict_put(input_dict=input_dict[key],
                             keys=keys[1:],
                             value=value)



def dict_hash(dictionary: Dict[str, Any]) -> str:
    import hashlib
    import json
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def dict_equal(*args):
    '''
    compares a list of dictionaries that are hasable by dict_hash
    '''

    if not all([ isinstance(arg, dict) for arg in args]):
        return False
    for i in range(len(args)):
        for j in range(len(args)):
            if dict_hash(args[i]) != dict_hash(args[j]):
                return False

    return True

# def dict_linear(input, key_path=[], linear_dict={}):
#     if isinstance(input, dict):
#         keys = list(input_dict.keys())
#     elif isinstance(input, list):
#         keys = list(range(len(input)))
#     else:
#         return input

#     if type(input) in [dict,list]:
#         for k in keys:
#             v= input[k]
#             linear_dict['.'.join(key_path)] = input
        




def flat2deep(flat_dict:dict):
    deep_dict = {}
    assert isinstance(flat_dict, dict)
    for k,v in flat_dict.items():
        dict_put(input_dict=deep_dict, keys=k, value=v)
    
    return deep_dict


def deep2flat(x:dict, root_key=None, flat_dict={}):
    new_flat_dict = {}
    if isinstance(x, dict):

        for k,v in x.items():
            new_root_key = k  if root_key == None else '.'.join([root_key, k])
            new_flat_dict[new_root_key] = deep2flat(x=v,  root_key = new_root_key, flat_dict=flat_dict)
    elif isinstance(x, list):
        for i,v in enumerate(x):
            new_root_key = str(i)  if root_key == None else '.'.join([root_key, str(i)])
            new_flat_dict[new_root_key] = deep2flat(x=v,  root_key = new_root_key, flat_dict=flat_dict)
    else:
        flat_dict[root_key] = x

    return flat_dict


def any_get(x:dict, keys:list , default=None):
    '''
    return x[k] for any of the list of 
    keys where k is an element in keys
    '''
    
    for k in keys:
        assert isinstance(k, str)
        output = x.get(k, None)
        if output != None:
            return output

    return default

dict_any = any_get




def dict_override(input_dict, override={}):      
    assert isinstance(override, dict), type(override)
    assert isinstance(input_dict, dict), type(input_dict)
    for k,v in override.items():
        dict_put(input_dict, k, v)

    return input_dict


def dict_merge(*args):
    output_dict = {}
    for arg in args:
        assert isinstance(arg, dict), f"{arg} is not a dict"
        output_dict.update(arg)
    return output_dict



async def async_get_json(path, return_type='dict', handle_error=True, default = None):
    from commune.utils.asyncio import async_read
    try: 
        data = await async_read(path)
        data = json.loads(await async_read(path))
    except FileNotFoundError as e:
        if handle_error:
            return default
        else:
            raise e

    if return_type in ['dict', 'json']:
        data = data
    elif return_type in ['pandas', 'pd']:
        data = pd.DataFrame(data)
    elif return_type in ['torch']:
        raise NotImplemented('Torch Not Implemented')
    return data


read_json = load_json = get_json = sync_wrapper(async_get_json)

async def async_put_json( path, data):
    
    from commune.utils.asyncio import  async_write
    # Directly from dictionary
    path = ensure_path(path)
    data_type = type(data)
    if data_type in [dict, list, tuple, set, float, str, int]:
        json_str = json.dumps(data)
    elif data_type in [pd.DataFrame]:
        json_str = json.dumps(data.to_dict())

    elif data_type in [np.ndarray]:
        json_str = json.dumps(data.tolist())
    elif data_type in [np.float32, np.float64, np.float16]:
        json_str = json.dumps(float(data))
    elif data_type in [Munch]:
        json_str = json.dumps(data.toDict())
    else:
        raise NotImplementedError(f"{data_type}, is not supported")
    
    return await async_write(path, json_str)

put_json = save_json = sync_wrapper(async_put_json)



async def async_get_yaml(path:str, return_type:str='dict', handle_error: bool = False):
    from commune.utils.asyncio import async_read
    
    try:  
        
        data = yaml.load(await async_read(path), Loader=yaml.Loader)
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
        raise NotImplemented('Torch not implemented')
    return data

read_yaml = load_yaml = get_yaml = sync_wrapper(async_get_yaml)

async def async_put_yaml( path, data):
    
    from commune.utils.asyncio import  async_write

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



from munch import Munch


def dict2munch(x:dict, recursive:bool=True)-> Munch:
    '''
    Turn dictionary into Munch
    '''
    if isinstance(x, dict):
        for k,v in x.items():
            if isinstance(v, dict) and recursive:
                x[k] = dict2munch(v)
        x = Munch(x)
    return x 

def munch2dict(x:Munch, recursive:bool=True)-> dict:
    '''
    Turn munch object  into dictionary
    '''
    if isinstance(x, Munch):
        x = dict(x)
        for k,v in x.items():
            if isinstance(v, Munch) and recursive:
                x[k] = munch2dict(v)

    return x 



def check_kwargs(kwargs:dict, defaults:Union[list, dict], return_bool=False):
    '''
    params:
        kwargs: dictionary of key word arguments
        defaults: list or dictionary of keywords->types
    '''
    try:
        assert isinstance(kwargs, dict)
        if isinstance(defaults, list):
            for k in defaults:
                assert k in defaults
        elif isinstance(defaults, dict):
            for k,k_type in defaults.items():
                assert isinstance(kwargs[k], k_type)
    except Exception as e:
        if return_bool:
            return False
        
        else:
            raise e

