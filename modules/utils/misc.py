


class SimpleNamespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

class RecursiveNamespace:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, dict):
            self.__dict__[k] = RecursiveNamespace(**v)

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
