import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):
    whitelist: List = ['put', 'get', 'get_hash']

    def __init__(self, max_replicas:int = 1, network='local', **kwargs):
        self.replica_map = {}
        self.max_replicas = max_replicas
        self.network = network
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()


    @property
    def store_dirpath(self) -> str:
        return self.resolve_path(f'{self.tag}.store')

    def resolve_store_path(self, key: str) -> str:
        path =  f'{self.store_dirpath}/{key}'
        return path

    def put(self, k,  v: Dict, timestamp:int=None, encrypt:bool=False, **kwargs):
        if timestamp == None:
            timestamp = c.timestamp()
        obj = {'data': v}
        v = self.serializer.serialize(obj)
        v = self.key.sign(v, return_json=True)

        path = self.resolve_store_path(k)
        return c.put(path, v)
    
    @property
    def peers(self):
        return [m for m in c.servers(self.module_path()) if m != self.server_name]
    

    def check_replicas(self):
        
        replicas = self.replicas
        c.print(replicas)
        c.print(replicas)
    

    def get(self,k, deserialize:bool= True) -> Any:
        path = self.resolve_store_path(k)
        v = c.get(path, {})
        if 'data' not in v:
            return {'success': False, 'error': 'No data found'}
        if deserialize:
            v = self.serializer.deserialize(v['data'])

        return v['data']
    
    
    def replicate(self, k, module) -> str:
        self.replicas

    


    def get_hash(self, k: str, seed : int= None , seed_sep:str = '<SEED>') -> str:
        obj = self.get(k, deserialize=False)
        if seed != None:
            obj = obj + seed_sep + str(seed)

        obj_hash = self.hash(obj, seed=seed)
        return c.hash(obj)

    def resolve_seed(self, seed: int = None) -> int:
        return c.timestamp() if seed == None else seed

    def remote_has(self, k: str, module: str, seed=None, **kwargs) -> bool:

        if isinstance(module, str):
            module = c.connect(module, **kwargs)
        
        seed = self.resolve_seed(seed)

        obj = self.get(k)
        obj['seed'] = seed
        local_hash = c.hash(obj)
        remote_hash =  module.get_hash(k, seed=seed)
        return bool(local_hash == remote_hash)
        

    def exists(self, k) -> bool:
        path = self.resolve_store_path(k)
        return c.exists(path)
    has = exists

    def rm(self, k) -> bool:
        assert self.exists(k), f'Key {k} does not exist'
        path = self.resolve_store_path(k)
        return c.rm(path)


    def ls_keys(self, search=None) -> List:
        path = self.store_dirpath
        path += f'/{search}' if search != None else ''
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
        import torch
        object_list = [0, {'fam': 1}, 'whadup', {'tensor': torch.rand(3,3)}, {'tensor': torch.rand(3,3), 'fam': 1}]
        for obj in object_list:
            c.print(f'putting {obj}')
            self.put('test', obj)
            get_obj = self.get('test', deserialize=False)
            obj_str = self.serializer.serialize(obj)

            # test hash
            assert self.get_hash('test', seed=1) == self.get_hash('test', seed=1)
            assert self.get_hash('test', seed=1) != self.get_hash('test', seed=2)
            assert obj_str == obj_str, f'Failed to put {obj} and get {get_obj}'






    