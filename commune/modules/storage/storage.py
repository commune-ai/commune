import commune
from typing import *
import streamlit as st

class Storage(commune.Module):

    def __init__(self, store: Dict = None, key: 'Key' = None):
        self.set_storage(store)
        self.set_key(key)
    
    def set_storage(self, storage: Dict = None):
        storage = {} if storage == None else storage
        assert isinstance(storage, dict), 'storage must be a dictionary'
        self.storage = storage
            
    def put(self,
            k:str, 
            v: Any,
            key: str = None,
            encrypt: bool = False,
            lifetime: bool = None) -> str:
        kwargs = c.locals2kwargs(locals())
        k =  kwargs.pop('k') # rm k
        self.storage[k] = kwargs
        return True
    
    def state_dict(self):
        import json
        state_dict = {}
        for k, v in self.storage.items():
            try:
                state_dict[k] = json.dumps(v)
            except:
                commune.log(f'could not serialize {k}')
            
        return state_dict
    def from_state_dict(self, state_dict: Dict) -> None:
        import json
        for k, v in state_dict.items():
            self.storage[k] = json.loads(v)
            
    def save(self, tag: str = None):
        tag = self.resolve_tag(tag)
        return self.put_json( path=tag, data=self.state_dict())

    def load(self, tag: str = None):
        tag = self.resolve_tag(tag)
        state_dict = self.get_json( path=tag)
        for k, v in state_dict.items():
            setattr(self, k, v)
    
    def get(self,
            k, 
            key:str = None,
            max_staleness: int = 1000) -> Any:
        key = c.resolve_key(key)
        item = self.storage[k]
        verified = key.verify(item)
        # decrypt if necessary
        if self.is_encrypted(item):
            item['data'] = key.decrypt(item['data'])
        item['data'] = self.str2python(item['data'])
        assert verified, 'could not verify signature'
        # check staleness
        staleness = commune.time() - item['data']['time']
        assert staleness < max_staleness

        return item['data']['data']

    @property
    def key2address(self) -> Dict:
        key2address = {}
        for k, v in self.storage.items():
            id = v['ss58_address']
            if id  in key2address:
                key2address[v['ss58_address']] += [k]
            else:
                key2address[v['ss58_address']] = [k]
        return key2address
        
    def is_encrypted(self, item: Dict) -> bool:
        return item.get('encrypt', False)
 
    @classmethod
    def test(cls):
        self = cls()
        object_list = [0, {'fam': 1}, 'whadup']
        for obj in object_list:
            self.put('test', obj)
            assert self.get('test') == obj
            
    