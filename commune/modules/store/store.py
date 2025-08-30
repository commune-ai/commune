

import commune as c
import os
from .utils import get_json
class Store:
    free = False
    endpoints = ['put', 'get']
    def __init__(self, path='~/.commune/module/'):
        self.path = os.path.abspath(os.path.expanduser(path))
    
    def resolve_path(self, path):
        return c.resolve_path(self.path + path)
    
    def item_path(self, item):
        return self.resolve_path(item + '.json')

    def put(self, k, v):
        k = self.item_path(k)
        return c.put_json(k, v)

    def get(self, k):
        k = self.item_path(k)
        return c.get_json(k)

    def rm(self, k):
        k = self.item_path(k)
        return c.rm(k)

    def ls(self, path = './'):
        return c.ls(self.resolve_path(path))

    def exists(self, path):
        return c.exists(self.resolve_path(path))

    def glob(self, path = './'):
        return c.glob(self.resolve_path(path))

    def hash(self, path):
        return c.hash(self.get(path))
    
    def test(self, key='test', value={'a':1}):
        self.put(key, value)
        assert self.get(key) == value
        assert c.hash(self.get(key)) == self.hash(key)
        self.rm(key)
        assert self.get('test') == None
        print('Store test passed')
        return {
            'status': 'pass'
        }


