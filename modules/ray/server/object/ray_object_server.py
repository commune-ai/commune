import ray
import os, sys
sys.path.append(os.getenv('PWD'))
from ray.util.queue import Queue
from commune import Module
"""

Background Actor for Message Brokers Between Quees

"""
from commune.block.ray.utils import kill_actor, create_actor
from commune.utils.dict import dict_put, dict_get



class ObjectServer(Module):
    default_config_path = 'ray.server.object'
    cache_dict= {}
    flat_cache_dict = {}

    def __init__(self, config=None):
        Module.__init__(self, config=config)

    def put(self,key,value):
        object_id = ray.put(value)
        self.cache_dict[key]= value
        return object_id
    def get(self, key, get=True):
        object_id= self.cache_dict.get(key)
        if isinstance(object_id, ray._raylet.ObjectRef):
            if get:
                return ray.get(object_id)
        return object_id

    def get_cache_state(self, key=''):
        object_id_list ={}
        key_path_list = {}
        return 

    @property
    def get_fn(fn):
        if isinstance(fn, str):
            fn =  eval(f'lambda x: {fn}')
        elif callable(fn):
            pass

        assert callable(fn)
        return fn

    def search(self, recursive=True, *args, **kwargs):
        object_dict =  {k:self.cache_dict[k]for k in self.search_keys(*args, **kwargs)}
        if recursive:
            new_object_dict = {}
            for k,v in object_dict.items():
                dict_put(input_dict=new_object_dict, keys=k, value=v)
            return new_object_dict
    def search_keys(self, key=None, filter_fn = None):
            
        if isinstance(key,str):
            filter_fn = lambda x: key in x
        elif callable(key):
            filter_fn = key
        elif key == None:
            filter_fn = lambda x: True


        return list(filter(filter_fn, list(self.cache_dict.keys())))


    def pop(self, *args ,**kwargs):
        keys = self.search(*args, **kwargs)

        pop_dict = {}
        for k in keys:
            pop_dict[k] = self.cache_dict.pop(k)
        
        if recursive:
            pop_dict = self.flat2deep
        return pop_dict
        object_id = dict_delete(input_dict=self.cache_dict, keys=key)

    def ls(self, key=''):
        return dict_get(input_dict=self.cache_dict, keys=key)
    def glob(self,key=''):
        return dict_get(input_dict=self.cache_dict, keys=key)


    def has(self, key):
        return dict_has(input_dict=self.cache_dict, keys=key)


if __name__ == "__main__":
    module = ObjectServerModule.deploy(actor=False, ray={'address': 'auto'})
    st.write(str(module.__class__).split('.')[-1]==str(module.get_object('ray.server.object.module.ObjectServerModule')).split('.')[-1])
    # st.write( module.__class__.__file__)
    ClientModule()
    st.write(str(module.get_object('ray.server.object.module.ObjectServerModule')))
    st.write(module.put('hey.fam.whdup', {'whadup.yo': 'fam'}))
    st.write(module.search())