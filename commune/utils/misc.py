

def cache(path='/tmp/cache.pkl', mode='memory'):

    def cache_fn(fn):
        def wrapped_fn(*args, **kwargs):
            cache_object = None
            self = args[0]

            
            if mode in ['local', 'local.json']:
                try:
                    cache_object = self.client.local.get_pickle(path, handle_error=False)
                except FileNotFoundError as e:
                    pass
            elif mode in ['memory', 'main.memory']:
                if not hasattr(self, '_cache'):
                    self._cache = {}
                else:
                    assert isinstance(self._cache, dict)
                cache_object = self._cache.get(path)
            force_update = kwargs.get('force_update', False)
            if not isinstance(cache_object,type(None)) or force_update:
                return cache_object
    
            cache_object = fn(*args, **kwargs)

            # write
            if mode in ['local']:

                st.write(cache_object)
                self.client.local.put_pickle(data=cache_object,path= path)
            elif mode in ['memory', 'main.memory']:
                '''
                supports main memory caching within self._cache
                '''
                self._cache[path] = cache_object
            return cache_object
        return wrapped_fn
    return cache_fn
    




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



from typing import *
import asyncio
from functools import partial
import random
import os
from copy import deepcopy
import concurrent

class Misc:

@staticmethod
def chunk(sequence:list = [0,2,3,4,5,6,6,7],
        chunk_size:int=4,
        num_chunks:int= None):
    assert chunk_size != None or num_chunks != None, 'must specify chunk_size or num_chunks'
    if chunk_size == None:
        chunk_size = len(sequence) / num_chunks
    if chunk_size > len(sequence):
        return [sequence]
    if num_chunks == None:
        num_chunks = int(len(sequence) / chunk_size)
    if num_chunks == 0:
        num_chunks = 1
    chunks = [[] for i in range(num_chunks)]
    for i, element in enumerate(sequence):
        idx = i % num_chunks
        chunks[idx].append(element)
    return chunks


def batch(cls, x: list, batch_size:int=8): 
    return cls.chunk(x, chunk_size=batch_size)

def cancel(self, futures):
    for f in futures:
        f.cancel()
    return {'success': True, 'msg': 'cancelled futures'}
    


def cachefn(cls, func, max_age=60, update=False, cache=True, cache_folder='cachefn'):
    import functools
    path_name = cache_folder+'/'+func.__name__
    def wrapper(*args, **kwargs):
        fn_name = func.__name__
        cache_params = {'max_age': max_age, 'cache': cache}
        for k, v in cache_params.items():
            cache_params[k] = kwargs.pop(k, v)

        
        if not update:
            result = cls.get(fn_name, **cache_params)
            if result != None:
                return result

        result = func(*args, **kwargs)
        
        if cache:
            cls.put(fn_name, result, cache=cache)
        return result
    return wrapper


@staticmethod
def round(x:Union[float, int], sig: int=6, small_value: float=1.0e-9):
    from commune.utils.math import round_sig
    return round_sig(x, sig=sig, small_value=small_value)


def round_decimals(cls, x:Union[float, int], decimals: int=6, small_value: float=1.0e-9):
    
    import math
    """
    Rounds x to the number of {sig} digits
    :param x:
    :param sig: signifant digit
    :param small_value: smallest possible value
    :return:
    """
    x = float(x)
    return round(x, decimals)




@staticmethod
def num_words( text):
    return len(text.split(' '))


def random_word(cls, *args, n=1, seperator='_', **kwargs):
    import commune as c
    random_words = cls.module('key').generate_mnemonic(*args, **kwargs).split(' ')[0]
    random_words = random_words.split(' ')[:n]
    if n == 1:
        return random_words[0]
    else:
        return seperator.join(random_words.split(' ')[:n])


def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
    return [text for text in text_list if filter_text in text]



@staticmethod
def tqdm(*args, **kwargs):
    from tqdm import tqdm
    return tqdm(*args, **kwargs)

progress = tqdm

emojis = {
    'smile': '😊',
    'sad': '😞',
    'heart': '❤️',
    'star': '⭐',
    'fire': '🔥',
    'check': '✅',
    'cross': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'question': '❓',
    'exclamation': '❗',
    'plus': '➕',
    'minus': '➖',

}



def emoji(cls, name:str):
    return cls.emojis.get(name, '❓')

@staticmethod
def tqdm(*args, **kwargs):
    from tqdm import tqdm
    return tqdm(*args, **kwargs)
progress = tqdm





def jload(cls, json_string):
    import json
    return json.loads(json_string.replace("'", '"'))


def partial(cls, fn, *args, **kwargs):
    return partial(fn, *args, **kwargs)
    
    

def sizeof(cls, obj):
    import sys
    sizeof = 0
    if isinstance(obj, dict):
        for k,v in obj.items():
            sizeof +=  cls.sizeof(k) + cls.sizeof(v)
    elif isinstance(obj, list):
        for v in obj:
            sizeof += cls.sizeof(v)
    elif any([k.lower() in cls.type_str(obj).lower() for k in ['torch', 'Tensor'] ]):

        sizeof += cls.get_tensor_size(obj)
    else:
        sizeof += sys.getsizeof(obj)
            
    return sizeof



def put_torch(cls, path:str, data:Dict,  **kwargs):
    import torch
    path = cls.resolve_path(path=path, extension='pt')
    torch.save(data, path)
    return path

def init_nn(self):
    import torch
    torch.nn.Module.__init__(self)



def locals2hash(self, kwargs:dict = {'a': 1}, keys=['kwargs']) -> str:
    kwargs.pop('cls', None)
    kwargs.pop('self', None)
    return self.dict2hash(kwargs)


def dict2hash(cls, d:dict) -> str:
    for k in d.keys():
        assert cls.jsonable(d[k]), f'{k} is not jsonable'
    return cls.hash(d)


def dict_put(cls, *args, **kwargs):
    from commune.utils.dict import dict_put
    return dict_put(*args, **kwargs)


def dict_get(cls, *args, **kwargs):
    from commune.utils.dict import dict_get
    return dict_get(*args, **kwargs)



def is_address(cls, address:str) -> bool:
    if not isinstance(address, str):
        return False
    if '://' in address:
        return True
    conds = []
    conds.append(len(address.split('.')) >= 3)
    conds.append(isinstance(address, str))
    conds.append(':' in address)
    conds.append(cls.is_int(address.split(':')[-1]))
    return all(conds)



def new_event_loop(cls, nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if nest_asyncio:
        cls.nest_asyncio()
    
    return loop


def set_event_loop(self, loop=None, new_loop:bool = False) -> 'asyncio.AbstractEventLoop':
    import asyncio
    try:
        if new_loop:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            loop = loop if loop else asyncio.get_event_loop()
    except RuntimeError as e:
        self.new_event_loop()
        
    self.loop = loop
    return self.loop


def get_event_loop(cls, nest_asyncio:bool = True) -> 'asyncio.AbstractEventLoop':
    try:
        loop = asyncio.get_event_loop()
    except Exception as e:
        loop = cls.new_event_loop(nest_asyncio=nest_asyncio)
    return loop



    

def merge(cls,  from_obj= None, 
                    to_obj = None,
                    include_hidden:bool=True, 
                    allow_conflicts:bool=True, 
                    verbose: bool = False):
    
    '''
    Merge the functions of a python object into the current object (a)
    '''
    from_obj = from_obj or cls
    to_obj = to_obj or cls
    
    for fn in dir(from_obj):
        if fn.startswith('_') and not include_hidden:
            continue
        if hasattr(to_obj, fn) and not allow_conflicts:
            continue
        if verbose:
            cls.print(f'Adding {fn}')
        setattr(to_obj, fn, getattr(from_obj, fn))
        
    return to_obj

    
# JUPYTER NOTEBOOKS

def enable_jupyter(cls):
    cls.nest_asyncio()



jupyter = enable_jupyter



def pip_list(cls, lib=None):
    pip_list =  cls.cmd(f'pip list', verbose=False, bash=True).split('\n')
    if lib != None:
        pip_list = [l for l in pip_list if l.startswith(lib)]
    return pip_list



def pip_libs(cls):
    return list(cls.lib2version().values())


def ensure_lib(cls, lib:str, verbose:bool=False):
    if  cls.pip_exists(lib):
        return {'lib':lib, 'version':cls.version(lib), 'status':'exists'}
    elif cls.pip_exists(lib) == False:
        cls.pip_install(lib, verbose=verbose)
    return {'lib':lib, 'version':cls.version(lib), 'status':'installed'}

required_libs = []

def ensure_libs(cls, libs: List[str] = None, verbose:bool=False):
    if hasattr(cls, 'libs'):
        libs = cls.libs
    results = []
    for lib in libs:
        results.append(cls.ensure_lib(lib, verbose=verbose))
    return results


def install(cls, libs: List[str] = None, verbose:bool=False):
    return cls.ensure_libs(libs, verbose=verbose)


def ensure_env(cls):
    cls.ensure_libs(cls.libs)

ensure_package = ensure_lib


def queue(cls, size:str=-1, *args,  mode='queue', **kwargs):
    if mode == 'queue':
        return cls.import_object('queue.Queue')(size, *args, **kwargs)
    elif mode in ['multiprocessing', 'mp', 'process']:
        return cls.module('process')(size, *args, **kwargs)
    elif mode == 'ray':
        return cls.import_object('ray.util.queue.Queue')(size, *args, **kwargs)
    elif mode == 'redis':
        return cls.import_object('redis.Queue')(size, *args, **kwargs)
    elif mode == 'rabbitmq':
        return cls.import_object('pika.Queue')(size, *args, **kwargs)
    else:
        raise NotImplementedError(f'mode {mode} not implemented')



def is_class(module: Any) -> bool:
    return type(module).__name__ == 'type' 



def param_keys(cls, model:'nn.Module' = None)->List[str]:
    model = cls.resolve_model(model)
    return list(model.state_dict().keys())


def params_map(cls, model, fmt='b'):
    params_map = {}
    state_dict = cls.resolve_model(model).state_dict()
    for k,v in state_dict.items():
        params_map[k] = {'shape': list(v.shape) ,
                            'size': cls.get_tensor_size(v, fmt=fmt),
                            'dtype': str(v.dtype),
                            'requires_grad': v.requires_grad,
                            'device': v.device,
                            'numel': v.numel(),
                            
                            }
        
    return params_map




def get_shortcut(cls, shortcut:str) -> dict:
    return cls.shortcuts().get(shortcut)


def rm_shortcut(cls, shortcut) -> str:
    shortcuts = cls.shortcuts()
    if shortcut in shortcuts:
        cls.shortcuts.pop(shortcut)
        cls.put_json('shortcuts', cls.shortcuts)
    return shortcut




def repo_url(cls, *args, **kwargs):
    return cls.module('git').repo_url(*args, **kwargs)    






def compose(cls, *args, **kwargs):
    return cls.module('docker').compose(*args, **kwargs)



def ps(cls, *args, **kwargs):
    return cls.get_module('docker').ps(*args, **kwargs)


def has_gpus(cls): 
    return bool(len(cls.gpus())>0)



def split_gather(cls,jobs:list, n=3,  **kwargs)-> list:
    if len(jobs) < n:
        return cls.gather(jobs, **kwargs)
    gather_jobs = [asyncio.gather(*job_chunk) for job_chunk in cls.chunk(jobs, num_chunks=n)]
    gather_results = cls.gather(gather_jobs, **kwargs)
    results = []
    for gather_result in gather_results:
        results += gather_result
    return results


def addresses(cls, *args, **kwargs) -> List[str]:
    return list(cls.namespace(*args,**kwargs).values())


def address_exists(cls, address:str) -> List[str]:
    addresses = cls.addresses()
    return address in addresses


    

def task(cls, fn, timeout=1, mode='asyncio'):
    
    if mode == 'asyncio':
        assert callable(fn)
        future = asyncio.wait_for(fn, timeout=timeout)
        return future
    else:
        raise NotImplemented
    


def shuffle(cls, x:list)->list:
    if len(x) == 0:
        return x
    random.shuffle(x)
    return x

@staticmethod
def detailed_error(e) -> dict:
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    file_name = tb[-1].filename
    line_no = tb[-1].lineno
    line_text = tb[-1].line
    response = {
        'success': False,
        'error': str(e),
        'file_name': file_name,
        'line_no': line_no,
        'line_text': line_text
    }   
    return response

def retry(fn, trials:int = 3, verbose:bool = True):
    # if fn is a self method, then it will be a bound method, and we need to get the function
    if hasattr(fn, '__self__'):
        fn = fn.__func__
    def wrapper(*args, **kwargs):
        for i in range(trials):
            try:
                print(fn)
                return fn(*args, **kwargs)
            except Exception as e:
                if verbose:
                    print(detailed_error(e))
                    print(f'Retrying {fn.__name__} {i+1}/{trials}')

    return wrapper


@staticmethod
def reverse_map(x:dict)->dict:
    '''
    reverse a dictionary
    '''
    return {v:k for k,v in x.items()}


def df(cls, x, **kwargs):
    return cls.import_object('pandas.DataFrame')(x, **kwargs)


def torch(cls):
    return cls.import_module('torch')


def tensor(cls, *args, **kwargs):
    return cls.import_object('torch.tensor')(*args, **kwargs)


@staticmethod
def random_int(start_value=100, end_value=None):
    if end_value == None: 
        end_value = start_value
        start_value, end_value = 0 , start_value
    
    assert start_value != None, 'start_value must be provided'
    assert end_value != None, 'end_value must be provided'
    return random.randint(start_value, end_value)



def mean(self, x:list=[0,1,2,3,4,5,6,7,8,9,10]):
    if not isinstance(x, list):
        x = list(x)
    return sum(x) / len(x)

def median(self, x:list=[0,1,2,3,4,5,6,7,8,9,10]):
    if not isinstance(x, list):
        x = list(x)
    x = sorted(x)
    n = len(x)
    if n % 2 == 0:
        return (x[n//2] + x[n//2 - 1]) / 2
    else:
        return x[n//2]


def stdev(cls, x:list= [0,1,2,3,4,5,6,7,8,9,10], p=2):
    if not isinstance(x, list):
        x = list(x)
    mean = cls.mean(x)
    return (sum([(i - mean)**p for i in x]) / len(x))**(1/p)
std = stdev


def set_env(cls, key:str, value:str)-> None:
    '''
    Pay attention to this function. It sets the environment variable
    '''
    os.environ[key] = value
    return value 



def pwd(cls):
    pwd = os.getenv('PWD', cls.libpath) # the current wor king directory from the process starts 
    return pwd


def choice(cls, options:Union[list, dict])->list:
    options = deepcopy(options) # copy to avoid changing the original
    if len(options) == 0:
        return None
    if isinstance(options, dict):
        options = list(options.values())
    assert isinstance(options, list),'options must be a list'
    return random.choice(options)


def sample(cls, options:list, n=2):
    if isinstance(options, int):
        options = list(range(options))
    options = cls.shuffle(options)
    return options[:n]
    

def chown(cls, path:str = None, sudo:bool =True):
    path = cls.resolve_path(path)
    user = cls.env('USER')
    cmd = f'chown -R {user}:{user} {path}'
    cls.cmd(cmd , sudo=sudo, verbose=True)
    return {'success':True, 'message':f'chown cache {path}'}


def chown_cache(cls, sudo:bool = True):
    return cls.chown(cls.cache_path, sudo=sudo)
    

def colors(cls):
    return ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'bright_black', 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white']
colours = colors

def random_color(cls):
    return random.choice(cls.colors())
randcolor = randcolour = colour = color = random_colour = random_color


def get_util(self, util:str):
    return self.get_module(util)


def random_float(cls, min=0, max=1):
    return random.uniform(min, max)


def random_ratio_selection(cls, x:list, ratio:float = 0.5)->list:
    if type(x) in [float, int]:
        x = list(range(int(x)))
    assert len(x)>0
    if ratio == 1:
        return x
    assert ratio > 0 and ratio <= 1
    random.shuffle(x)
    k = max(int(len(x) * ratio),1)
    return x[:k]


def link_cmd(cls, old, new):
    
    link_cmd = cls.get('link_cmd', {})
    assert isinstance(old, str), old
    assert isinstance(new, str), new
    link_cmd[new] = old 
    
    cls.put('link_cmd', link_cmd)



            

def resolve_memory(cls, memory: Union[str, int, float]) -> str:
                
    scale_map = {
        'kb': 1e3,
        'mb': 1e6,
        'gb': 1e9,
        'b': 1,
    }
    if isinstance(memory, str):
        scale_found = False
        for scale_key, scale_value in scale_map.items():
            
            
            if isinstance(memory, str) and memory.lower().endswith(scale_key):
                memory = int(int(memory[:-len(scale_key)].strip())*scale_value)
                

            if type(memory) in [float, int]:
                scale_found = True
                break
                
    assert type(memory) in [float, int], f'memory must be a float or int, got {type(memory)}'
    return memory
        



def filter(cls, text_list: List[str], filter_text: str) -> List[str]:
    return [text for text in text_list if filter_text in text]



def is_success(cls, x):
    # assume that if the result is a dictionary, and it has an error key, then it is an error
    if isinstance(x, dict):
        if 'error' in x:
            return False
        if 'success' in x and x['success'] == False:
            return False
        
    return True


def is_error(cls, x:Any):
    """
    The function checks if the result is an error
    The error is a dictionary with an error key set to True
    """
    if isinstance(x, dict):
        if 'error' in x and x['error'] == True:
            return True
        if 'success' in x and x['success'] == False:
            return True
    return False


def is_int(cls, value) -> bool:
    o = False
    try :
        int(value)
        if '.' not in str(value):
            o =  True
    except:
        pass
    return o

    

def is_float(cls, value) -> bool:
    o =  False
    try :
        float(value)
        if '.' in str(value):
            o = True
    except:
        pass

    return o 




def timer(cls, *args, **kwargs):
    from commune.utils.time import Timer
    return Timer(*args, **kwargs)


def timeit(cls, fn, *args, include_result=False, **kwargs):

    t = cls.time()
    if isinstance(fn, str):
        fn = cls.get_fn(fn)
    result = fn(*args, **kwargs)
    response = {
        'latency': cls.time() - t,
        'fn': fn.__name__,
        
    }
    if include_result:
        print(response)
        return result
    return response

@staticmethod
def remotewrap(fn, remote_key:str = 'remote'):
    '''
    calls your function if you wrap it as such

    @c.remotewrap
    def fn():
        pass
        
    # deploy it as a remote function
    fn(remote=True)
    '''

    def remotewrap(self, *args, **kwargs):
        remote = kwargs.pop(remote_key, False)
        if remote:
            return self.remote_fn(module=self, fn=fn.__name__, args=args, kwargs=kwargs)
        else:
            return fn(self, *args, **kwargs)
    
    return remotewrap


@staticmethod
def is_mnemonic(s: str) -> bool:
    import re
    # Match 12 or 24 words separated by spaces
    return bool(re.match(r'^(\w+ ){11}\w+$', s)) or bool(re.match(r'^(\w+ ){23}\w+$', s))

@staticmethod   
def is_private_key(s: str) -> bool:
    import re
    # Match a 64-character hexadecimal string
    pattern = r'^[0-9a-fA-F]{64}$'
    return bool(re.match(pattern, s))



@staticmethod
def address2ip(address:str) -> str:
    return str('.'.join(address.split(':')[:-1]))

@staticmethod
def as_completed( futures, timeout=10, **kwargs):
    return concurrent.futures.as_completed(futures, timeout=timeout, **kwargs)



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



def munch(cls, x:Dict) -> 'Munch':
    '''
    Converts a dict to a munch
    '''
    return cls.dict2munch(x)


  
def time( cls, t=None) -> float:
    import time
    if t is not None:
        return time.time() - t
    else:
        return time.time()


def datetime(cls):
    import datetime
    # UTC 
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H:%M:%S")


def time2datetime(cls, t:float):
    import datetime
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d_%H:%M:%S")

time2date = time2datetime


def datetime2time(cls, x:str):
    import datetime
    return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()

date2time =  datetime2time


def delta_t(cls, t):
    return t - cls.time()

def timestamp(cls) -> float:
    return int(cls.time())

def sleep(cls, seconds:float) -> None:
    import time
    time.sleep(seconds)
    return None


def search_dict(self, d:dict = 'k,d', search:str = {'k.d': 1}) -> dict:
    search = search.split(',')
    new_d = {}

    for k,v in d.items():
        if search in k.lower():
            new_d[k] = v
    
    return new_d


def path2text(cls, path:str, relative=False):

    path = cls.resolve_path(path)
    assert os.path.exists(path), f'path {path} does not exist'
    if os.path.isdir(path):
        filepath_list = cls.glob(path + '/**')
    else:
        assert os.path.exists(path), f'path {path} does not exist'
        filepath_list = [path] 
    path2text = {}
    for filepath in filepath_list:
        try:
            path2text[filepath] = cls.get_text(filepath)
        except Exception as e:
            pass
    if relative:
        pwd = cls.pwd()
        path2text = {os.path.relpath(k, pwd):v for k,v in path2text.items()}
    return path2text


def root_key(cls):
    return cls.get_key()


def root_key_address(cls) -> str:
    return cls.root_key().ss58_address



def is_root_key(cls, address:str)-> str:
    return address == cls.root_key().ss58_address


@staticmethod
def repo2module( repo, module = None):
    if module == None:
        module = os.path.basename(repo).replace('.git','').replace(' ','_').replace('-','_').lower()
    
    cls.new_module(module=module, repo=repo)
    return {'module':module, 'repo':repo, 'status':'success'}

# time within the context

def context_timer(cls, *args, **kwargs):
    return cls.timer(*args, **kwargs)



def folder_structure(cls, path:str='./', search='py', max_depth:int=5, depth:int=0)-> dict:
    import glob
    files = cls.glob(path + '/**')
    results = []
    for file in files:
        if os.path.isdir(file):
            cls.folder_structure(file, search=search, max_depth=max_depth, depth=depth+1)
        else:
            if search in file:
                results.append(file)

    return results



def copy(cls, data: Any) -> Any:
    import copy
    return copy.deepcopy(data)
