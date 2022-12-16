from .main import *
from .namespace import * 
from .shell import *
from .process import *
from .function import *
from .time import *
from .object import *
from .json import *
from .asyncio import *
from .yaml import *


from typing import Union
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
    
