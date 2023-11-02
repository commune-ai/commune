import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):
    whitelist: List = ['put', 'get', 'item_hash']

    def __init__(self, 
                 max_replicas:int = 1, 
                network='local',
                validate:bool = False,
                match_replica_prefix : bool = False,
                **kwargs):
        self.replica_map = {}
        self.max_replicas = max_replicas
        self.network = network
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()
        self.executor = c.module('executor')()
        if validate:
            self.match_replica_prefix = match_replica_prefix
            c.thread(self.validate)

    @property
    def store_dirpath(self) -> str:
        tag = self.tag
        if tag == None:
            tag = 'base'
        return self.resolve_path(f'store/{tag}')

    def resolve_store_path(self, key: str) -> str:
        path =  f'{self.store_dirpath}/{key}'
        return path
    

    def num_files(self) -> int:
        return len(self.files)
    
    def files(self) -> List:
        return c.ls(self.store_dirpath)
    
    def item2info(self, search=None):
        files = self.files()
        item2info = {}
        for file in files:
            name = file.split('/')[-1].split('.')[0]
            file_info = {
                'size': c.format_data_size(c.filesize(file), fmt='b'),
                'path': file,
            }
            if search != None:
                if search not in name:
                    continue
            item2info[name] = file_info
        
        return item2info
    @property
    def items(self, search=None) -> List:
        items =  list(self.item2info(search=search).keys())
        return items
    def file2size(self, fmt:str='b') -> int:
        files = self.files()
        file2size = {}
        for file in files:
            file2size[file] = c.format_data_size(c.filesize(file), fmt=fmt)
        return file2size
    
    def resolve_key(self, key=None) -> str:
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key

    def put_item(self, k,  v: Dict, encrypt:bool=False, replicas = 1, key=None):
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
        peer.put_item(k, v)
        replica_map[k] = [peer]

    def get_item(self,k, deserialize:bool= True, key=None) -> Any:
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

    def item_hash(self, k: str = None, seed : int= None , seed_sep:str = '<SEED>', obj=None) -> str:
        if obj == None:
            assert k != None, 'Must provide k or obj'
            obj = self.get_item(k, deserialize=False)
        if seed != None:
            obj = str(obj) + seed_sep + str(seed)
        return self.hash(obj, seed=seed)

    def resolve_seed(self, seed: int = None) -> int:
        return c.timestamp() if seed == None else seed

        
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
                self.put_item('test', obj,encrypt=encrypt)
                get_obj = self.get('test', deserialize=False)
                obj_str = self.serializer.serialize(obj)

                # test hash
                assert self.item_hash('test', seed=1) == self.item_hash('test', seed=1)
                assert self.item_hash('test', seed=1) != self.item_hash('test', seed=2)
                assert obj_str == obj_str, f'Failed to put {obj} and get {get_obj}'

                self.rm('test')
            
    @classmethod
    def test_verify(cls):
        import torch
        storage_modules = [cls() for i in range(10)]
        object_list = [0, {'fam': 1}, 'whadup', {'tensor': torch.rand(3,3)}, {'tensor': torch.rand(3,3), 'fam': 1}]
        for i, x in enumerate(object_list):
            for i, storage in enumerate(storage_modules):
                storage.put_item('test', x)
            seed = c.time()
            for i, storage_i in enumerate(storage_modules):
                for j, storage_j in enumerate(storage_modules):
                    c.print(f'Verifying i={i} j={j}')
                    assert storage_i.item_hash('test', seed=seed) == storage_j.item_hash('test', seed=seed)


    item2replicas = {}
    def validate(self, refresh=False):

        self.ls()
        item2info = self.item2info()
        item_keys = list(item2info.keys())
        item_key = c.choice(item_keys)
        item = self.get_item(item_key)
        item_hash = c.hash(item)
        remote_item_key = f'replica::{item_hash}'

        # get the peers 
        peers = self.peers()
        max_replicas = min(self.max_replicas, len(peers)) 
        # get the replica_map
        replica_map = {} if refresh else self.get(f'replica_map/{self.tag}',{})
        replica_peers = self.item2replicas.get(item_key, [])
        has_enough_replicas = bool(len(replica_peers) >= max_replicas)
        if has_enough_replicas: 
            # check if replicas match
            peer = c.choice(replica_peers)
            seed = c.timestamp()
            # get the local hash
            local_hash = self.item_hash(obj=item, seed=seed)

            # check if remote hash matches
            remote_hash= c.connect(peer).item_hash(remote_item_key, seed=seed)
            if local_hash != remote_hash:
                # remove replica
                replica_peers.remove(peer)
                c.print(f'Hashes do not match for {item_key} on {peer}', color='red')
            else:
                c.print(f'Hashes match for {item_key} on {peer}', color='green')
                
        else:
            # find peer to add replica
            candidate_peers = [peer for peer in peers if peer not in replica_peers]
            peer = c.choice(candidate_peers)
            # add replica
            response = c.call(peer, 'put', remote_item_key, item)

            if response['success']:
                replica_peers += [peer]
                c.print(f'Added replica for {item_key} on {peer}', color='green')
            else: 
                c.print(f'Failed to add replica for {item_key} on {peer}', color='red')

        replica_map[item_key] = replica_peers
        self.put(f'replica_map/{self.tag}', replica_map)

        return {'success': True, 'replica_map': replica_map}