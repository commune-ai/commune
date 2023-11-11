import commune as c
from typing import *
import streamlit as st
import json

class Storage(c.Module):
    whitelist: List = ['put_item', 'get_item', 'hash_item', 'items']
    replica_prefix = 'replica::'
    shard_prefix = 'shard::'

    def __init__(self, 
                 max_replicas:int = 2, 
                network='local',
                validate:bool = False,
                match_replica_prefix : bool = False,
                max_shard_size = 200,
                tag = None,
                **kwargs):
        self.network = network
        self.max_shard_size = max_shard_size
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

    

    def store_dir(self, tag=None) -> str:
        tag = self.resolve_tag(tag)
        if tag == None:
            tag = 'base'
        return self.resolve_path(f'{tag}/store')

    def resolve_item_path(self, key: str, tag=None) -> str:
        store_dir = self.store_dir(tag=tag)
        if not key.startswith(store_dir):
            key = f'{store_dir}/{key}'
        return key
    
    def files(self, tag=None) -> List:
        return sorted(c.ls(self.store_dir(tag=tag)))
    
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

    def peers(self,) -> List:
        return c.namespace(self.module_path(), network=self.network)

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

    def put_metadata(self, k:str, metadata:Dict, tag=None):
        assert self.item_exists(k, tag=tag), f'Key {k} does not exist with {tag}'
        k = self.resolve_item_path(k, tag=tag)
        path = k + '/metadata.json'
        return self.put_json(path, metadata)

    def get_metadata(self, k, tag=None):
        k = self.resolve_item_path(k, tag=tag)
        path = k + '/metadata.json'
        return c.get_json(path, default={})  
    
    def get_item_replicas(self, k:str, tag:str=None) -> List[str]:
        metadata = self.get_metadata(k, tag=tag)
        return metadata.get('replicas', [])
    
    def put_metadata(self, k, metadata:Dict, tag=None):
        assert self.item_exists(k, tag=tag), f'Key {k} does not exist with {tag}'
        k = self.resolve_item_path(k, tag=tag)
        path = k + '/metadata.json'
        return self.put_json(path, metadata)
    
    def refresh_store(self, tag=None):
        tag = self.resolve_tag(tag)
        path = self.store_dir(tag=tag)
        return c.rm(path)

    def put_item(self, k,  v: Dict, encrypt:bool=False,  tag=None, serialize:bool = True):
        timestamp = c.timestamp()
        k = self.resolve_item_path(k, tag=tag)    
        path = {
            'data': k +'/data.json',
            'metadata': k + '/metadata.json'
        }    

        data = v

        if serialize:
            data = self.serializer.serialize(data)

        if isinstance(data, str):
            size_bytes = len(data) 
        else:
            size_bytes = c.sizeof(data)
        c.print(f'Putting {k} with {size_bytes} bytes', color='green')
        is_shard = bool(self.shard_prefix in k)

        shards = []
        if size_bytes > self.max_shard_size and not is_shard:               
            # split it along the bytes
            # round up to the nearest shard size
            num_shards = (size_bytes // self.max_shard_size) + (1 if size_bytes % self.max_shard_size != 0 else 0)
            for i in range(num_shards):
                shard_path = f'{k}/{self.shard_prefix}{i}'
                # split it along the bytes
                shard = data[i*self.max_shard_size:(i+1)*self.max_shard_size]
                assert len(shard) <= self.max_shard_size, f'Shard must be less than {self.max_shard_size} bytes, got {len(shard)}'
                self.put_item(k=shard_path, v=shard, encrypt=encrypt, tag=tag, serialize=False)
                shards += [shard_path]

        else:

            self.put_json(path['data'], data)
            # encrypt it if you want
            if encrypt:
                data = self.key.encrypt(data)   
            # sign it for verif
            data = self.key.sign(data, return_json=True)

       

        # SAVE METADATA 
        metadata = {
            'size_bytes': size_bytes,
            'timestamp': timestamp,
            'encrypt': encrypt,
            'key': self.key.ss58_address ,
            'path': path,
            'shards': shards
        }

        self.put_json(path['metadata'], metadata)

        return {'success': True, 'key': k,  'metadata': metadata}
    

    def rm_item(self, k):
        k = self.resolve_item_path(k)
        return c.rm(k)



    def rm_items(self, search):
        items = self.items(search=search)
        for item in items:
            self.rm_item(item)
        return {'success': True, 'items': items}
    

    def drop_peers(self, item_key):
        metadata = self.get_metadata(item_key)
        return peers



    def get_item(self,k:str, deserialize:bool= True, include_metadata=False) -> Any:
        k = self.resolve_item_path(k)
        metadata = self.get_json(k+'/metadata.json', {})

        shards = metadata.get('shards', [])
        if len(shards) > 0:
            data = ''
            for shard_path in metadata['shards']:
                c.print(f'Getting shard {shard_path}')
                shard = self.get_item(shard_path, deserialize=False)
                c.print(f'Got shard {shard_path}, {shard}')
                data += shard
            data = self.serializer.deserialize(data)
        else:
            data = self.get_json(k+'/data.json', {})


        if isinstance(data, dict):
            if 'data' not in data:
                return {'success': False, 'error': 'No data found'}
            if 'encrypted' in metadata and metadata['encrypted']:
                data['data'] = self.key.decrypt(data['data'])

            if deserialize:
                data['data'] = self.serializer.deserialize(data['data'])

            data = data['data']

        # include
        if include_metadata:
            data['metadata'] = metadata
            return data
        else:
            return data
    

    def hash_item(self, k: str = None, seed : int= None , seed_sep:str = '<SEED>', data=None) -> str:
        """
        Hash a string
        """
        if data == None:
            assert k != None, 'Must provide k or obj'
            data = self.get_item(k, deserialize=False)
        if seed != None:
            data = str(data) + seed_sep + str(seed)
        return self.hash(data, seed=seed)

    def item_exists(self, k, tag=None) -> bool:
        path = self.resolve_item_path(k, tag=tag)
        return c.exists(path)
    has = exists = item_exists

    def rm(self, k , tag=None) -> bool:
        assert self.exists(k, tag=tag), f'Key {k} does not exist with {tag}'
        path = self.resolve_item_path(k, tag=tag)
        return c.rm(path)
    

    def item_paths(self, tag=None):
        sore_dir = self.store_dir(tag=tag)
        return [x for x in c.ls(sore_dir)]

    def items(self, search=None, include_replicas:bool=False, tag=None) -> List:
        """
        List the item names
        """
        path = self.store_dir(tag=tag)
        items = [x.split('/')[-1] for x in c.ls(path)]
        
        if search != None:
            items = [x for x in items if search in x]

        if include_replicas == False:
            items = [x for x in items if not x.startswith(self.replica_prefix)]

        return items
    
    def replica_items(self, tag:str=None) -> List:
        return [x for x in self.items(tag=tag) if x.startswith(self.replica_prefix)]
        

    def refresh(self, tag:str=None) -> None:
        path = self.store_dir(tag=tag)
        return c.rm(path)
    
    def validate(self, item_key:str = None):
        item_key = c.choice(self.items()) if item_key == None else item_key
        item_data = self.get_item(item_key)
        metadata = self.get_metadata(item_key)
        shards = self.get_shards(item_key, metadata=metadata)

        if len(shards) > 0:
            responses = []
            for shard_item in shards:
                responses += [self.validate(shard_item)]
            return responses
        # get the peers 
        peers = list(self.peers().keys())
        max_replicas = min(self.max_replicas, len(peers)) 
        replica_peers = metadata.get('replicas', [])
        has_enough_replicas = bool(len(replica_peers) >= max_replicas)

        # get the remote replica key
        remote_item_key = f'{self.replica_prefix}/{c.hash(item_data)}'



        if has_enough_replicas: 
            # check if replicas match
            peer = c.choice(replica_peers)

            # use the timestamp for the check seed
            seed = c.timestamp()

            # get the local hash
            local_hash = self.hash_item(data=item_data, seed=seed)
            # check if remote hash matches
            success_remote_hash = False
            try:
                c.print(f'Checking {item_key} on {peer}')
                remote_hash = c.call(peer, 'hash_item', remote_item_key, seed=seed)
                success_remote_hash = bool(local_hash == remote_hash)
            except Exception as e:
                c.print(e)
                c.print(f'Failed to get remote hash for {item_key} on {peer}', color='red')
                remote_hash = None

            if success_remote_hash:
                # remove replica from the registry
                replica_peers.remove(peer)
                c.print(f'Hashes do not match for {item_key} on {peer}', color='red')
            else:
                # Dope, the pass had checked
                c.print(f'Hashes match for {item_key} on {peer}', color='green')
                
        # now check it again 
        if not has_enough_replicas:
            # find peer to add replica and add it
            candidate_peers = [peer for peer in peers if peer not in replica_peers]
            peer = c.choice(candidate_peers)
            
            # add replica
            response = c.call(peer, 'put_item', remote_item_key, item_data)
            c.print(response)
            if 'error' in response:
                c.print(f'Failed to add replica for {item_key} on {peer} ', color='red')
            else:
                # dope no
                replica_peers += [peer]
                c.print(f'Added replica for {item_key} on {peer}', color='green')
        
        metadata['replicas'] = replica_peers
        self.put_metadata(item_key, metadata)

        return {'success': True, 'metadata': metadata, 'msg': f'Validated {item_key}'}
   

    def validate_loop(self, tag=None, interval=1.0, vote_inteval=1, init_timeout = 10):
        c.sleep(init_timeout)
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
                k = 'test'
                self.put_item(k, obj,encrypt=encrypt)
                assert self.item_exists(k), f'Failed to put {obj}'
                get_obj = self.get_item(k, deserialize=False)
                c.print(f'putting {k} and got {get_obj}')

                obj_str = self.serializer.serialize(get_obj)
                assert obj_str == self.serializer.serialize(get_obj), f'Failed to put {obj} and get {get_obj}'
                # test hash
                assert self.hash_item(k, seed=1) == self.hash_item(k, seed=1)
                assert self.hash_item(k, seed=1) != self.hash_item(k, seed=2)
                assert obj_str == obj_str, f'Failed to put {obj} and get {get_obj}'

                # self.rm(k)

                # assert not self.item_exists(k)

        return {'success': True, 'msg': 'its all done fam'}


    def get_shards(self, k:str, tag=None, metadata=None) -> List:
        metadata = self.get_metadata(k, tag=tag) if metadata == None else metadata
        store_dir = self.store_dir(tag=tag) 
        shards = metadata.get('shards', [])
        return [x.replace(store_dir, '') for x in shards]


    def put_dummies(self, tag=None):
        tag = self.resolve_tag(tag)
        for i in range(10):
            k = f'test{i}'
            self.put_item(k, {'dummy': i})
            assert self.item_exists(k), f'Failed to put {k}'

