import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):

    def __init__(self, store: Dict = None, **kwargs):
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()
    @property
    def store_dirpath(self) -> str:
        tag = self.resolve_tag()
        return self.resolve_path(f'{tag}.store')

    def resolve_store_path(self, key: str) -> str:
        path =  f'{self.store_dirpath}/{key}'
        return path

    def put(self, k,  v: Dict, tag: str = None):
        tag = self.resolve_tag(tag)
        v = self.serializer.serialize({'data': v})
        v = self.key.sign(v, return_json=True)
        path = self.resolve_store_path(k)
        return c.put(path, v)

    def get(self,k) -> Any:
        path = self.resolve_store_path(k)
        v = c.get(f'{self.store_dirpath}/{k}', {})
        v = v['data']
        v = self.serializer.deserialize(v)['data']
        return v

    def exists(self, k) -> bool:
        path = self.resolve_store_path(k)
        c.print(path)
        return c.exists(path)

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
            
    