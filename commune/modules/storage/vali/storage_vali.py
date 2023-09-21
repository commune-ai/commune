import commune as c
from typing import *


class StorageVali(c.Module):
    def __init__(self, config=None, **kwargs):
        config = self.set_config(config, kwargs=kwargs)
        history = {}

        self.storage = c.module('storage')()

    storage_peers = {}

    def get_storage_peer_info(self, name:str) -> Dict:
        default_storage_peer = {'address': None,  'size': 0, 'stored_keys': [], 'w': 0}
        return self.get(f'{self.tag}/peers/{name}', default_storage_peer)
    
    

    def set_storage_peer_info(self, name:str, value:dict):
        assert isinstance(value, dict)
        assert 'address' in value
        assert 'size' in value
        assert 'stored_keys' in value
        assert 'w' in value

        return self.put(f'{self.tag}/peers/{name}', value)

    def score_module(self, module, **kwargs) -> float:



        info = module.info()
        key = self.storage.next_key()
        
        obj = self.storage.get(key, deserialize=False)

        obj_size = c.sizeof(obj)
        remote_obj_key = c.hash(obj)
        remote_has = self.storage.remote_has(remote_obj_key, module=module)
        if not remote_has:
            module.put(key, obj)
        
        storage_peer = self.get_storage_peer(info['name'])

        # does the module have the key
        remote_has = self.storage.remote_has(remote_obj_key, module=module)


        # if the module has the key, then add it to the storage_peer
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        else:
            module.put(remote_obj_key, obj, timeout=self.config.storage_timeout)
            remote_has = self.storage.remote_has(remote_obj_key, module=module)
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        else:

            storage_peer['size'] -= obj_size
            storage_peer['stored_keys'] = [k for k in storage_peer['stored_keys'] if k != key]
        
        # ensure no duplicates
        storage_peer['stored_keys'] = list(set(storage_peer['stored_keys']))
        storage_peer['size'] = max(storage_peer['size'], 0)

        # set weight
        storage_peer['w'] = storage_peer['size']
        self.set_storage_peer(info['name'], storage_peer)
        
        return storage_peer
        


    def put(self, *args,**kwargs):
        return self.storage.put(*args,**kwargs)

        
    def get(self, *args,**kwargs):
        return self.storage.get(*args,**kwargs)


    @classmethod
    def test(cls, n=10):
        storage = [c.module('storage')() for i in range(n)]
        self = cls()

        self.score_module(storage[0])

