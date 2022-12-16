
from commune.utils import dict_any

def cache(self, **kwargs):

    if 'keys' in kwargs:
        key_dict = {k: kwargs.get('keys') for k in ['save', 'load']}
    else:
        key_dict = dict(save=dict_any(x=input_kwargs, keys=['save', 'write'], default=[]),
                        load= dict_any(x=input_kwargs, keys=['load', 'read'], default=[]))

    def wrap_fn(fn):
        def new_fn(self, *args, **kwargs):
            [setattr(self, k, self.get_json(k)) for k in key_dict['load']]
            fn(self, *args, **kwargs)
            [self.put_json(k, getattr(self, k)) for k in key_dict['save']]
        return new_fn
    
    return wrap_fn
def enable_cache(**input_kwargs):
    load_kwargs = dict_any(x=input_kwargs, keys=['load', 'read'], default={})
    if isinstance(load_kwargs, bool):
        load_kwargs = dict(enable=load_kwargs)

    save_kwargs = dict_any(x=input_kwargs, keys=['save', 'write'], default={})
    if isinstance(save_kwargs, bool):
        save_kwargs = dict(enable=save_kwargs)

    refresh = dict_any(x=input_kwargs, keys=['refresh', 'refresh_cache'], default=False)
    assert isinstance(refresh, bool), f'{type(refresh)}'

    def wrapper_fn(fn):
        def new_fn(self, *args, **kwargs):

            if refresh: 
                self.cache = {}
            else:
                self.load_cache(**load_kwargs)

            output = fn(self, *args, **kwargs)
            self.save_cache(**save_kwargs)
            return output
        
        return new_fn
    return wrapper_fn
