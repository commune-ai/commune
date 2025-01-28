
import commune as c
from typing import *
import pandas as pd

class History(c.Module):
    def __init__(self,
                  path='history',
                  max_age=100000,
                   **kwargs):
        self.max_age = max_age        
        self.set_history_path(path)
    # HISTORY 
    def check_item(self, item, required_fields=['address', 'timestamp']):
        assert all([field in item for field in required_fields]), f'Missing required fields: {required_fields}'
        assert c.valid_ss58_address(item['address']), f'Invalid address: {item["address"]}'
    
    def get_user_directory(self, key):
        key_address = c.resolve_key_address(key)
        return self.history_path + '/' + key_address
    
    def get_user_path(self, key_address):
        if not c.valid_ss58_address(key_address):
            key_address = c.get_key(key_address).ss58_address
        path = self.history_path +f'/{key_address}/{c.time()}.json'
        return path

    def refresh_history(self):
        path = self.history_path
        self.rm(path)
        return c.ls(path)

    def add_history(self, item):
        self.check_item(item)
        path = self.get_user_path(item['address'])
        if 'path' in item:
            path = item['path']
        self.put(path, item)
        return {'path': path, 'item': item}
    
    def rm_history(self, key):
        path = self.get_user_directory(key)
        self.rm(path)
        return {'path': path}
    
    def history_size(self, key):
        path = self.get_user_directory(key)
        return len(c.ls(path))
    
    def history_exists(self, key):
        path = self.get_user_directory(key)
        return self.exists(path) and self.history_size(key) > 0

    def user_history(self, key):
        path = self.get_user_directory(key)
        return c.ls(path)
    def set_history_path(self, path):
        self.history_path = self.resolve_path(path)
        return {'history_path': self.history_path}
    

    def test_history(self):
        key = c.new_key()
        item = {'address': key.ss58_address, 'timestamp': c.time()}
        self.add_history(item)
        assert self.history_exists(key.ss58_address)
        self.user_history(key.ss58_address)
        self.rm_history(key.ss58_address)
        assert not self.history_exists(key.ss58_address)
        return {'key': key.ss58_address, 'item': item}

