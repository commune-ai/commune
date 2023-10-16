import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):
    whitelist: List = ['put', 'get', 'get_hash']

    def __init__(self, max_replicas:int = 1, network='local',**kwargs):
        self.replica_map = {}
        self.max_replicas = max_replicas
        self.network = network
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()
        self.executor = c.module('executor')()



    @property
    def store_dirpath(self) -> str:
        tag = self.tag
        if tag == None:
            tag = 'base'
        return self.resolve_path(f'{tag}.store')

    def resolve_store_path(self, key: str) -> str:
        path =  f'{self.store_dirpath}/{key}'
        return path

    def resolve_key(self, key=None) -> str:
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key

    def put(self, k,  v: Dict, encrypt:bool=False, replicas = 1, key=None):
        timestamp = c.timestamp()
        obj = v

        k = self.resolve_store_path(k)
        # serialize
        v = self.serializer.serialize(obj)
        if encrypt:
            v = self.key.encrypt(v)
        v = self.key.sign(v, return_json=True)
        v['encrypted'] = encrypt
        v['timestamp'] = timestamp
        
        if replicas > 1:
            self.replicate(k, v, replicas=replicas)
        self.put_json(k, v)
        size_bytes = self.sizeof(v)
        return {'success': True, 'key': k,  'size_bytes': size_bytes, 'replica_map': self.replica_map}
    
    def replicate(self, k, v, replicas=2):
        replica_map = self.get('replica_map', default={})
        peer = self.random_peer()
        peer.put(k, v)
        replica_map[k] = [peer]




    def check_replicas(self):
        
        replicas = self.replicas
        c.print(replicas)
        c.print(replicas)
    

    def get(self,k, deserialize:bool= True, key=None) -> Any:
        k = self.resolve_store_path(k)
        v = self.get_json(k, {})

        if 'data' not in v:
            return {'success': False, 'error': 'No data found'}
        c.print(v)
        if 'encrypted' in v and v['encrypted']:
            c.print(v)
            v['data'] = self.key.decrypt(v['data'])


        if deserialize:
            v['data'] = self.serializer.deserialize(v['data'])
        return v['data']
    
    
    def replicate(self, k, module) -> str:
        self.replicas

    def get_hash(self, k: str, seed : int= None , seed_sep:str = '<SEED>') -> str:
        obj = self.get(k, deserialize=False)
        c.print(obj)
        if seed != None:
            obj = str(obj) + seed_sep + str(seed)
        return self.hash(obj, seed=seed)

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

    def items(self, search=None) -> List:
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
    def cachefn(cls, func, max_age=60, update=False, cache=True, cache_folder='cachefn'):
        import functools
        path_name = cache_folder+'/'+func.__name__
        def wrapper(*args, **kwargs):
            fn_name = func.__name__
            cache_params = {'max_age': max_age, 'cache': cache}
            for k, v in cache_params.items():
                cache_params[k] = kwargs.pop(k, v)
            if not update:
                result = cls.get(fn_name, default=None, **cache_params)
                if result != None:
                    return result

            result = func(*args, **kwargs)
            
            if cache:
                cls.put(fn_name, result, cache=cache)

            return result

        return wrapper
        
    @classmethod
    def test(cls):
        c.print('STARTING')
        self = cls()
        import torch
        object_list = [0, {'fam': 1}, 'whadup', {'tensor': torch.rand(3,3)}, {'tensor': torch.rand(3,3), 'fam': 1}]

        for encrypt in [True, False]:
            for obj in object_list:
                c.print(f'putting {obj}')
                self.put('test', obj,encrypt=encrypt)
                get_obj = self.get('test', deserialize=False)
                obj_str = self.serializer.serialize(obj)

                # test hash
                assert self.get_hash('test', seed=1) == self.get_hash('test', seed=1)
                assert self.get_hash('test', seed=1) != self.get_hash('test', seed=2)
                assert obj_str == obj_str, f'Failed to put {obj} and get {get_obj}'

                self.rm('test')
            
    @classmethod
    def test_verify(cls):
        storage_modules = [cls() for i in range(10)]
        object_list = [0, {'fam': 1}, 'whadup', {'tensor': torch.rand(3,3)}, {'tensor': torch.rand(3,3), 'fam': 1}]
        for i, x in enumerate(object_list):
            for i, storage in enumerate(storage_modules):
                storage.put('test', x)
            seed = c.time()
            for i, storage_i in enumerate(storage_modules):
                for j, storage_j in enumerate(storage_modules):
                    c.print(f'Verifying i={i} j={j}')
                    assert storage_i.get_hash('test', seed=seed) == storage_j.get_hash('test', seed=seed)

