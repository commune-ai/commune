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
        if timestamp == None:
            timestamp = c.timestamp()
        obj = {
            'data': v,
            'timestamp': timestamp,
            'hash': c.hash(v),
            'size': c.sizeof(v),
        }
        v = self.serializer.serialize(obj)
        v = self.key.sign(v, return_json=True)
        path = self.resolve_store_path(k)
        return c.put(path, v)

    

    def get(self,k, deserialize:bool= True) -> Any:
        path = self.resolve_store_path(k)
        c.print(f'getting {k} from {path}')
        v = c.get(path, {})
        if 'data' not in v:
            return {'success': False, 'error': 'No data found'}
        if deserialize:
            v = self.serializer.deserialize(v['data'])
        return v['data']



    def get_hash(self, k: str, seed : int= None ) -> str:
        obj = self.get(k)
        obj['seed'] = self.resolve_seed(seed)
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
            get_obj = self.get('test')
            get_obj_str = self.serializer.serialize(obj)
            obj_str = self.serializer.serialize(obj)
            assert obj_str == obj_str, f'Failed to put {obj} and get {get_obj}'

    storage_peers = {}


    def get_storage_peer(self, name, default=None) -> Dict:
        if default == None:
            default = {'address': None,  'size': 0, 'stored_keys': [], 'w': 0}
        return self.get(f'{self.tag}/peers/{name}', {})


    def score_module(self, module) -> float:


        info = module.info()
        obj_keys = self.ls_keys()
        key = c.choice(obj_keys)
        
        obj = self.get(key)

        obj_size = c.sizeof(obj)

        remote_has = self.remote_has(remote_obj_key, module=module)
        if not remote_has:
            module.put(key, obj)
        

        storage_peer = self.get_storage_peer(info['name'])

        remote_has = self.remote_has(remote_obj_key, module=module)
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        
        if not remote_has:
            remote_obj_key = obj['hash']
            module.put(remote_obj_key, obj)
            remote_has = self.remote_has(remote_obj_key, module=module)
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        else:

            storage_peer['size'] -= obj_size
            storage_peer['size'] = max(storage_peer['size'], 0)
            storage_peer['stored_keys'] = [k for k in storage_peer['stored_keys'] if k != key]

        # set weight
        storage_peer['w'] = storage_peer['size']
        

        return storage_peer
        






    