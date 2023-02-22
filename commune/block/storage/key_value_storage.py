
import commune
from typing import *

class KeyValueStorage(commune.Module):
    
    def __init__(self, store: Dict = None):

        
    def set_storage(self, storage: Dict):
        if storage is not None:
            assert isinstance(storage, dict), 'storage must be a dictionary'
        
        self.storage = storage
            
    def put(self, key:str, value: Any) -> str:
        self.storage[key] = value
        return key
        
    def get(self, key:str) -> Any:
        return self.storage[key]
    
    