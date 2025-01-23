

import commune as c
class Storage(c.Module):
    def __init__(self, path='storage'):
        self.path = self.resolve_path(path)
    def put(self, k, v):
        k = self.path + '/' + k
        return c.put(k, v)
    def get(self, k):
        k = self.path + '/' + k
        return c.get(k)
