import commune as c

reducer = c.module('reduce')
class Memory:

    def __init__(self, size=10000):
        self.size = size
        self.data = {}

    def add(self, key, value):
        self.data[key] = value

    def search(self, query=None):
        keys = list(self.data.keys())
        reducer.forward(keys, query=query)

    
    
