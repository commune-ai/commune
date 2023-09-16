import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):
    whitelist: List = ['put', 'get']

    def __init__(self, store: Dict = None, **kwargs):
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()
    @property
    def store_dirpath(self) -> str:
        return self.resolve_path(f'{self.tag}.store')

    def resolve_store_path(self, key: str) -> str:
        path =  f'{self.store_dirpath}/{key}'
        return path

    def put(self, k,  v: Dict, timestamp:int=None, ):
        if module != None:
            self = c.module(module)

        if timestamp == None:
            timestamp = c.timestamp()
        v = self.serializer.serialize({'data': v, 'timestamp': timestamp})
        v = self.key.sign(v, return_json=True)
        path = self.resolve_store_path(k)
        return c.put(path, v)

    

    def get(self,k) -> Any:
        path = self.resolve_store_path(k)
        c.print(f'getting {k} from {path}')
        v = c.get(path, {})
        if 'data' not in v:
            return {'success': False, 'error': 'No data found'}
        v = self.serializer.deserialize(v['data'])
        return v



    def get_hash(self, k: str, seed : int= None ) -> str:
        obj = self.get(k)
        obj['seed'] = self.resolve_seed(seed)
        return c.hash(obj)

    def resolve_seed(self, seed: int = None) -> int:
        return c.timestamp() if seed == None else seed

    def remote_obj_exists(self, k: str, module: str, seed=None, **kwargs) -> bool:
        storage = c.connect(module, **kwargs)
        seed = self.resolve_seed(seed)

        obj = self.get(k)
        obj['seed'] = seed
        return storage.get_hash(k, seed=seed)

    def exists(self, k) -> bool:
        path = self.resolve_store_path(k)
        return c.exists(path)
    has = exists

    def rm(self, k) -> bool:
        assert self.exists(k), f'Key {k} does not exist'
        path = self.resolve_store_path(k)
        return c.rm(path)


    def ls(self, search=None) -> List:
        path = self.store_dirpath
        return c.ls(path)


    def refresh(self) -> None:
        path = self.store_dirpath
        return c.rm(path)


    
    @property
    def key2address(self) -> Dict:
        key2address = {}
        for k, v in self.store.items():
            id = v['ss58_address']
            if id  in key2address:
                key2address[v['ss58_address']] += [k]
            else:
                key2address[v['ss58_address']] = [k]
        return key2address
        
    @classmethod
    def test(cls):
        c.print('STARTING')
        self = cls()
        object_list = [0, {'fam': 1}, 'whadup']
        for obj in object_list:
            c.print(f'putting {obj}')
            self.put('test', obj)
            assert self.get('test') == obj



    