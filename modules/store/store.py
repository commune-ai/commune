

import commune as c
class Store(c.Module):
    fns = ['put', 'get']
    def __init__(self, path='storage'):
        self.path = self.resolve_path(path)
    
    def resolve_path(self, path):
        return c.resolve_path('~/.commune/storage/' + path)
    
    def get_item_path(self, item):
        return self.resolve_path(item + '.json')

    def put(self, k, v):
        k = self.get_item_path(k)
        return c.put_json(k, v)

    def get(self, k):
        k = self.get_item_path(k)
        return c.get_json(k)

    def rm(self, k):
        k = self.get_item_path(k)
        return c.rm(k)

    def ls(self, path):
        return c.ls(self.resolve_path(path))

    def items(self, path='./'):
        return c.ls(self.resolve_path(path))

    def glob(self, path = './'):
        return c.glob(self.resolve_path(path))

    def hash(self, path):
        return c.hash(self.get(path))
    
    def test(self):
        self.put('test', {'a':1})
        assert self.get('test') == {'a':1}
        c.hash(self.get('test')) == self.hash('test')
        self.rm('test')
        assert self.get('test') == None
        print('Store test passed')
        return {'status':'pass'}


