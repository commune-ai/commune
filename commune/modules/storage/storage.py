import commune as c
from typing import *
import os

class Storage(c.Module):
    whitelist: List = ['put_item', 'get_item', 'hash', 'items']

    def __init__(self, store_dir = 'store', **kwargs):
        self.set_store_dir(store_dir)
        self.serializer = c.module('serializer')()

    def set_store_dir(self, store_dir):
        self.store_dir = self.resolve_path(store_dir or 'store')

    def resolve_item_path(self, path: str) -> str:
        store_dir = self.store_dir
        path = path if  path.startswith(store_dir) else f'{store_dir}/{path}' 
        path = self.resolve_path(path)
        return path
    
    def files(self) -> List:
        return sorted(c.ls(self.store_dir))
    
    def file2size(self, fmt:str='b') -> int:
        files = self.files()
        file2size = {}
        for file in files:
            file2size[file] = c.format_data_size(c.filesize(file), fmt=fmt)
        return file2size

    @c.endpoint()
    def put_item(self, k,  v: Dict, encrypt:bool=False):
        timestamp = c.timestamp()
        path = self.resolve_item_path(k)    
        # encrypt it if you want
        if encrypt:
            data = self.key.encrypt(data)   
        # sign it for verif
        v = self.key.sign(v, return_json=True)
        self.put_json(path, v)
        return {'success': True, 'path': path, 'timestamp': timestamp}
    
    @c.endpoint()
    def get_item(self,k:str) -> Any:
        k = self.resolve_item_path(k)
        v = self.get_json(k, {})
        if 'ticket' in v:
            c.ticket2address(v['ticket'])
        v = v['data']
        v = self.serializer.deserialize(v) 
        return v
    
    
    def hash_item(self, k: str = None, seed : int= None , seed_sep:str = '<SEED>', data=None, tag=None) -> str:
        """
        Hash a string
        """
        data = self.get_item(k)
        if seed != None:
            data = f'{data}{seed_sep}{seed}'
        return c.hash(data)

    def exists(self, k) -> bool:
        path = self.resolve_item_path(k)
        return c.exists(path)
    
    def test(self):
        results = []
        results.append(self.test_storage())
        results.append(self.test_hash())
        return results
    
    def rm_item(self, k):
        k = self.resolve_item_path(k)
        return c.rm(k)

    def rm_many(self, search):
        items = self.items(search=search)
        for item in items:
            self.rm_item(item)
        return {'success': True, 'items': items}
    

    def test_storage(self):
        key = 'test'
        value = {'test': 'value'}
        self.put_item(key, value)
        new_value = self.get_item(key)
        assert value == new_value, f'Error: {value} != {new_value}'
        return {'success': True, 'msg': 'Storage test passed'}
    
    def test_hash(self):
        k = 'test'
        value = 'value'
        self.put_item(k, value)
        hash1 = self.hash_item(k ,seed=1)
        hash2 = self.hash_item(k,  seed=1)
        assert hash1 == hash2, f'Error: {hash1} != {hash2}'

        hash3 = self.hash_item(k, seed=2)
        assert hash1 != hash3, f'Error: {hash1} == {hash3}'

        return {'success': True, 'msg': 'Storage hash test passed'}
    
    def items(self, search=None) -> List:
        """
        List the item names
        """
        store_dir = self.store_dir

        items = [x.split('/')[-1] for x in c.ls(store_dir)]
        
        if search != None:
            items = [x for x in items if search in x]

        return items
    



Storage.run(__name__)