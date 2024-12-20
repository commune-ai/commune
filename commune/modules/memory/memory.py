
import json


class Memory:
    def __init__(self, max_bytes_size = 100000):
        self.memory = []
        self.max_bytes_size = max_bytes_size
    

    def current_size(self):
        return sum([len(x) for x in self.memory])
    
    def check_storage(self, value:str):
        return len(self.memory) + len(value) <= self.max_bytes

    def store(self, value):
        value = json.dumps(value)
        self.memory += [value_hash]

    def load(self, address):
        return self.memory.get(address, 0)