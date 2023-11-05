import commune as c
from typing import *
import streamlit as st

class Storage(c.Module):
    whitelist: List = ['put_item', 'get_item', 'item_hash']

    def __init__(self, 
                 max_replicas:int = 2, 
                network='local',
                validate:bool = True,
                match_replica_prefix : bool = False,
                tag = None,
                **kwargs):
        self.max_replicas = max_replicas
        self.network = network
        self.set_config(kwargs=locals()) 
        self.serializer = c.module('serializer')()
        self.executor = c.module('executor')()
        if validate:
            self.match_replica_prefix = match_replica_prefix
            c.thread(self.validate_loop)


    def resolve_tag(self, tag=None):
        tag = tag if tag != None else self.tag
        return tag
    

    def store_dirpath(self, tag=None) -> str:
        tag = self.resolve_tag(tag)
        if tag == None:
            tag = 'base'
        return self.resolve_path(f'store/{tag}')

    def resolve_item_path(self, key: str, tag=None) -> str:
        path =  f'{self.store_dirpath(tag=tag)}/{key}'
        return path
    

    def num_files(self) -> int:
        return len(self.files)
    
    def files(self, tag=None) -> List:
        files = c.ls(self.store_dirpath(tag=tag))
        return sorted(files)
    
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

    def put_item(self, k,  v: Dict, encrypt:bool=False, replicas = 1, tag=None):
        timestamp = c.timestamp()
        k = self.resolve_item_path(k, tag=tag)    
        path = {
            'data': k +'/data',
            'metadata': k + '/metadata'
        }    
        # serialize
        data = self.serializer.serialize(v)

        # encrypt it if you want
        if encrypt:
            data = self.key.encrypt(data)
            
        # sign it for verif
        data = self.key.sign(data, return_json=True)

        c.makedirs(k, exist_ok=True)
        self.put_json(path['data'], data)


        # SAVE METADATA 
        metadata = {
            'size_bytes': self.sizeof(data),
            'timestamp': timestamp,
            'encrypt': encrypt,
            'key': self.key.ss58_address ,
            'size_bytes': c.format_data_size(c.filesize(path['data']), fmt='b'),
            'path': path
        }

        self.put_json(path['metadata'], metadata)

        size_bytes = self.sizeof(v)
        return {'success': True, 'key': k,  'metadata': metadata}
    
    def replicate(self, k, v, replicas=2):
        replica_map = self.get('replica_map', default={})
        peer = self.random_peer()
        peer.put_item(k, v)
        replica_map[k] = [peer]

    def rm_item(self, k):
        k = self.resolve_item_path(k)
        return c.rm(k)
    
    def rm_items(self, search):
        items = self.items(search=search)
        for item in items:
            self.rm_item(item)
        return {'success': True, 'items': items}

    def get_item(self,k, deserialize:bool= True, key=None) -> Any:
        k = self.resolve_item_path(k)
        data = self.get_json(k+'/data', {})
        metadata = self.get_json(k+'/metadata', {})
        if 'data' not in data:
            return {'success': False, 'error': 'No data found'}
        if 'encrypted' in metadata and metadata['encrypted']:
            data['data'] = self.key.decrypt(data['data'])

        if deserialize:
            data['data'] = self.serializer.deserialize(data['data'])
        return data['data']
    
    def get_metadata(self, k, tag=None):
        return c.get_json(self.resolve_item_path(k, tag=tag)+'/metadata')  

    def item_hash(self, k: str = None, seed : int= None , seed_sep:str = '<SEED>', obj=None) -> str:
        if obj == None:
            assert k != None, 'Must provide k or obj'
            obj = self.get_item(k, deserialize=False)
        if seed != None:
            obj = str(obj) + seed_sep + str(seed)
        return self.hash(obj, seed=seed)

    def resolve_seed(self, seed: int = None) -> int:
        return c.timestamp() if seed == None else seed

        
    def exists(self, k, tag=None) -> bool:
        path = self.resolve_item_path(k, tag=tag)
        return c.exists(path)
    has = exists

    def rm(self, k , tag=None) -> bool:
        assert self.exists(k, tag=tag), f'Key {k} does not exist with {tag}'
        path = self.resolve_item_path(k, tag=tag)
        return c.rm(path)
    

    def item_paths(self, tag=None):
        path = self.store_dirpath(tag=tag)
        return [x for x in c.ls(path)]

    def items(self, search=None, include_replicas:bool=False, tag=None) -> List:
        path = self.store_dirpath(tag=tag)
        items = [x.split('/')[-1] for x in c.ls(path)]
        
        if search != None:
            items = [x for x in items if search in x]

        if include_replicas == False:
            items = [x for x in items if "replica::" not in x]

        return items
    
    def replicas(self, tag=None) -> List:
        return self.items(search='replica::', tag=tag)
        

    def refresh(self, tag=None) -> None:
        path = self.store_dirpath(tag=tag)
        return c.rm(path)


    def replica_map(self):
        return self.get(f'replica_map/{self.tag}', default={})
    
    def set_replica_map(self, value):
        self.put(f'replica_map/{self.tag}', value)

    item2replicas = {}
    def validate(self, refresh=False):


        # get the item2info
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
        replica_peers = replica_map.get(item_key, [])
        has_enough_replicas = bool(len(replica_peers) >= max_replicas)
        
        if has_enough_replicas: 
            # check if replicas match
            peer = c.choice(replica_peers)
            seed = c.timestamp()
            # get the local hash
            local_hash = self.item_hash(obj=item, seed=seed)

            # check if remote hash matches
            remote_hash= c.call(peer, 'item_hash', remote_item_key, seed=seed)

            if local_hash != remote_hash:
                # remove replica
                replica_peers.remove(peer)
                c.print('local_hash', local_hash)
                c.print('remote_hash', remote_hash)
                c.print(f'Hashes do not match for {item_key} on {peer}', color='red')
            else:
                c.print(f'Hashes match for {item_key} on {peer}', color='green')
                
        else:
            # find peer to add replica
            candidate_peers = [peer for peer in peers if peer not in replica_peers]
            peer = c.choice(candidate_peers)
            # add replica
            response = c.call(peer, 'put_item', remote_item_key, item)
            c.print(response, '')

            if 'error' in response:
                c.print(f'Failed to add replica for {item_key} on {peer} ', color='red')
            else:
                replica_peers += [peer]
                c.print(f'Added replica for {item_key} on {peer}', color='green')

        replica_map[item_key] = replica_peers
        self.put(f'replica_map/{self.tag}', replica_map)

        return {'success': True, 'replica_map': replica_map}
    

    def validate_loop(self, tag=None, interval=0.1, vote_inteval=1):
        import time
        tag = self.tag if tag == None else tag
        while True:
            try:
                self.validate()
                time.sleep(interval)
            except Exception as e:
                c.print(e, color='red')


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
