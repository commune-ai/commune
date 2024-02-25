import commune as c
from typing import *
import streamlit as st
import json

class Storage(c.Module):
    whitelist: List = ['put_item', 'get_item', 'hash_item', 'items']
    replica_prefix = 'replica'
    shard_prefix = 'shard::'



    def __init__(self, 
                validate:bool = False,
                max_shard_size:str = 200,
                min_check_interval:str = 100,
                tag = None,
                **kwargs):
        
        config = self.set_config(kwargs=locals()) 

        self.network = config.network
        self.max_shard_size = config.max_shard_size
        self.max_replicas = config.max_replicas 
        self.min_check_interval = config.min_check_interval       
        self.serializer = c.module('serializer')()

        if validate:
            self.match_replica_prefix = match_replica_prefix
            c.thread(self.validate_loop)


    def resolve_item_path(self, k:str) -> str:
        return self.store_dir() + '/' + k

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

    def peers(self) -> List:
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



    def put(self, k,  
            v: Dict, 
            password:str=None, 
            ticket = None,
            timestamp=None, module=None):
        if module == None:
            kwargs = locals()
            kwargs.pip('module')
            module = c.connect(module).put(**kwargs)

        path = self.resolve_item_path(k) 

        timestamp = timestamp or c.timestamp()
        
        if password != None:
            v = c.encrypt(v, password=password)
            encrypted = True
        else:
            encrypted = False

        data = {
            'data': v,
            'timestamp': timestamp,
            'encrypted': encrypted,
        }
        data['signature'] = self.key.sign(data)

        self.check_data(data)

        self.put_json(path, data)
        
        return {'success': True, 'msg': f'Put {k} with {len(v)} bytes'}
    



    def check_data(self, data:Dict) -> bool:
        assert isinstance(data, dict), f'Data must be a dict, got {type(data)}'
        assert 'data' in data, f'Data must have a data key'
        assert 'timestamp' in data, f'Data must have a timestamp key'
        assert 'signature' in data, f'Data must have a signature key'
        

    def get(self,k:str, default=None, tag=None, password=None, raw:bool = False) -> Any:
        
        path = self.resolve_item_path(k, tag=tag)
        data = self.get_json(path, default=default)
        self.check_data(data)
        if password != None:
            data = c.decrypt(data['data'], password=password)
        if raw:
            return data
        return data['data']
    
    def exists(self, k, tag=None) -> bool:
        path = self.resolve_item_path(k, tag=tag)
        return c.exists(path)
    has = exists 

    def rm(self, k , tag=None) -> bool:
        assert self.exists(k, tag=tag), f'Key {k} does not exist with {tag}'
        path = self.resolve_item_path(k, tag=tag)
        return c.rm(path)
    

    def store_dir(self, tag=None) -> str:
        tag = self.tag if tag == None else tag
        return self.resolve_path(tag)
    
    def paths(self, tag=None):
        sore_dir = self.store_dir(tag=tag)
        return [x for x in c.ls(sore_dir)]

    def items(self, search=None, include_replicas:bool=True, tag=None) -> List:
        """
        List the item names
        """
        store_dir = self.store_dir(tag=tag)


        items = [x.split('/')[-1] for x in c.ls(store_dir)]
        
        if search != None:
            items = [x for x in items if search in x]

        return items
    
    def replica_items(self) -> List:
        return [x for x in self.items() if x.startswith(self.replica_prefix)]
        
    

    def refresh(self) -> None:
        path = self.store_dir()
        return c.rm(path)


    def validate_loop(self, interval=0.1, vote_inteval=1, init_timeout = 1):
        c.sleep(init_timeout)
        import time
        tag = self.tag if tag == None else tag
        while True:
            try:
                items = self.items()
                if len(items) == 0:
                    c.print('No items to validate', color='red')
                    time.sleep(1.0)
                    continue
                item_key = c.choice(items)
                metadata = self.get_metadata(item_key)
                num_replicas = len(metadata.get('replicas', []))
                c.print(f'Validating {item_key} key with {num_replicas} replicas -->', color='green')
                self.validate(item_key)
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
                assert self.exists(k), f'Failed to put {obj}'
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


    @classmethod
    def dashboard(cls):
        st.write('Storage')


Storage.run(__name__)