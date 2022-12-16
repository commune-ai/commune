import os
import time
from time import gmtime, strftime
import random
import yaml
from copy import deepcopy
import numpy as np
from contextlib import contextmanager
import torch
from importlib import import_module
import pickle
import math
import datetime

def round_sig(x, sig=6, small_value=1.0e-9):
    """
    Rounds x to the number of {sig} digits
    :param x:
    :param sig: signifant digit
    :param small_value: smallest possible value
    :return:
    """
    return round(x, sig - int(math.floor(math.log10(max(abs(x), abs(small_value))))) - 1)


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_yaml(fn):
    with open(fn, "r") as stream:
        cfg = yaml.load(stream)

    return cfg

def load_pickle(file_path, verbose=True):

    with open(file_path, 'rb') as f:
        object = pickle.load(f)
    if verbose:
        print("Loaded: ", file_path)
    return object


def dump_pickle(object, file_path, verbose=True):
    ensure_dir(file_path=file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)
    if verbose:
        print("Saved: ", file_path)


        
class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value * count
        self.count = count

    def update(self, value, count=1):
        self.total_value += value * count
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)



def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + datetime.timedelta(hours=t.minute // 30))


def check_distributions(kwargs):
    return {k: {"mean": round(v.double().mean().item(), 2), "std": round(v.double().std().item(), 2)} for k, v in
            kwargs.items() if isinstance(v, torch.Tensor)}

def seed_everything(seed: int) -> None:
    "seeding function for reproducibility"
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_current_time():
    return strftime("%m%d%H%M%S", gmtime())



@contextmanager
def timer(name: str) -> None:

    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


chunk_list = lambda my_list, n: [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]


def confuse_gradients(model):
    """

    :param model: model
    :return:
    """
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data = torch.randn(p.grad.data.shape).to(p.grad.data.device)


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

get_module_file = get_module



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

def torch_batchdictlist2dict(batch_dict_list, dim=0):
    """
    converts
        batch_dict_list: dictionary (str, tensor)
        to
        out_batch_dict : dictionary (str,tensor)

    along dimension (dim)

    """
    out_batch_dict = {}
    for batch_dict in batch_dict_list:

        for k, v in batch_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if k in out_batch_dict:
                out_batch_dict[k].append(v)
            else:
                out_batch_dict[k] = [v]

    # stack
    return {k: torch.cat(v, dim=dim) for k, v in out_batch_dict.items()}


def tensor_dict_shape(input_dict):
    out_dict = {}

    """should only have tensors/np.arrays in leafs"""
    for k,v in input_dict.items():
        if isinstance(v,dict):
            out_dict[k] = tensor_dict_shape(v)
        elif type(v) in [torch.Tensor, np.ndarray]:
            out_dict[k] = v.shape

    return out_dict


def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def equal_intervals_pandas_series(series, nbins=10):
    max = series.max()
    min = series.min()

    bin_size = (max - min) / nbins

    for bin_id in range(nbins):
        bin_bounds = [min + bin_id * bin_size,
                      min + (bin_id + 1) * bin_size]
        series = series.apply(lambda x: bin_bounds[0] if x >= bin_bounds[0] and x < bin_bounds[1] else x)

    return series


def nan_check(input, key_list=[], root_key=''):
    if isinstance(input, dict):
        for k, v in input.items():

            new_root_key = '.'.join([root_key, k])
            if type(v) in [dict, list]:
                nan_check(input=v,
                                    key_list=key_list,
                                    root_key=new_root_key)
            else:
                if isinstance(v, torch.Tensor):
                    if any(torch.isnan(v)):
                        key_list.append(new_root_key)
                else:
                    if math.isnan(v):
                        key_list.append(new_root_key)
    elif isinstance(input, list):
        for k, v in enumerate(input):
            new_root_key = '.'.join([root_key, str(k)])
            if type(v) in [dict, list]:
                nan_check(input=v,
                                    key_list=key_list,
                                    root_key=new_root_key)
            else:
                if isinstance(v, torch.Tensor):
                    if any(torch.isnan(v)):
                        key_list.append(new_root_key)
                else:
                    if math.isnan(v):
                        key_list.append(new_root_key)
    return key_list


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
  
def dict_put(input_dict,keys, value ):
    """
    insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    key = keys[0]
    if len(keys) == 1:
        if  isinstance(input_dict,dict):
            input_dict[key] = value

    elif len(keys) > 1:
        if key not in input_dict:
            input_dict[key] = {}
        dict_put(input_dict=input_dict[key],
                             keys=keys[1:],
                             value=value)

def dict_put(input_dict,keys, value ):
    """
    insert keys that are dot seperated (key1.key2.key3) recursively into a dictionary
    """
    if isinstance(keys, str):
        keys = keys.split('.')
    key = keys[0]
    if len(keys) == 1:
        if  isinstance(input_dict,dict):
            input_dict[key] = value

    elif len(keys) > 1:
        if key not in input_dict:
            input_dict[key] = {}
        dict_put(input_dict=input_dict[key],
                             keys=keys[1:],
                             value=value)



from typing import Dict, Any
import hashlib
import json

def dict_hash(dictionary: Dict[str, Any]) -> str:
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

    if isinstance(x, dict):

        for k,v in x.items():
            new_root_key = k  if root_key == None else '.'.join(root_key, k)
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


def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True



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

