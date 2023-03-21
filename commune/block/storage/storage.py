
import commune
from typing import *

class Storage(commune.Module):
    
    def __init__(self, store: Dict = None):
        self.set_storage(store)
        
    def set_storage(self, storage: Dict = None):
        storage = {} if storage == None else storage
        
        assert isinstance(storage, dict), 'storage must be a dictionary'
        
        self.storage = storage
            
    def put(self, key:str, value: Any, meta = None) -> str:
        self.storage[key] = {
            'data': value,
            'meta': meta if meta != None else {},
            
        }
        return key
    
    def state_dict(self):
        import json
        state_dict = {}
        for k, v in self.storage.items():
            try:
                state_dict[k] = json.dumps(v)
            except:
                pass
                commune.log(f'could not serialize {k}')
            
        
    def from_state_dict(self, state_dict: Dict) -> None:
        import json
        for k, v in state_dict.items():
            self.storage[k] = json.loads(v)
            
            
            
    def save(self, keys=None, path=None, password=None):
        if keys == None:
            keys = self.storage.keys()
        
        if path == None:
            path = self.get_path()
        
        if password == None:
            password = self.get_password()
        
        if password != None:
            for key in keys:
                self.encrypt(key, password)
        
        state_dict = self.state_dict()
        
        return self.put_json( path=path, data=state_dict)
    
    
    def load(self, path=None, password=None):
        
    
    
    def get(self, key:str, return_data: bool = True) -> Any:
        storage_object = self.storage[key]
        if return_data:
            storage_object = storage_object['data']
        return storage_object

    def get_aes_key(self, password: str) -> 'Key':
        if not hasattr(self, 'aes_key_class'):
            self.aes_key_class = commune.get_module('crypto.key.aes')
        return self.aes_key_class(password)
        
    def set_aes_key(self, password):
        self.aes_key = self.get_aes_key(password)
        
    def resolve_aes_key(self, password):
        if not hasattr(self, 'aes_key'):
            self.set_aes_key(password)
        return self.aes_key
        
    def encrypt(self, key:str, password: str) -> str:
        aes_key = self.get_aes_key(password)
        self.storage[key] = aes_key.encrypt(self.storage[key])
        
    def is_encrypted(self, key:str) -> bool:
        return isinstance(self.storage[key], bytes)
    
    def decrypt(self, key:str, password: str) -> str:
        # encrypt the the content of the key
        
        assert self.is_encrypted(key), 'key is not encrypted'
        
        aes_key = self.get_aes_key(password)
        self.storage[key] = aes_key.decrypt(self.storage[key])
    
    
    @classmethod
    def test(cls):
        self = cls()
        
        object_list = [0, {'fam': 1}, 'whadup']
        for obj in object_list:
            self.put('test', obj)
            assert self.get('test') == obj
            
        
    
if __name__ == "__main__":
    Storage.test()
    
    