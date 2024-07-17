import os
import commune as c

class ApiManager(c.Module):
    def __init__(self, path='api_vault', password=None, api_keys=[]):
        self.path = self.resolve_path(path)
        self.password = password

    @property
    def api_keys(self):
        return self.get(self.path, {})

    def add_api_key(self, name , api_key):
        api_keys = self.api_keys
        api_keys[name] = list(set(api_keys.get(name, []) + [api_key]))
        num_keys = len(api_keys[name])
        assert isinstance(api_keys, dict), api_keys
        self.save_api_keys(api_keys)
        return {'msg': f'api_key {name} added', 'num_keys': num_keys}
    
    def pop_api_key(self, name, index=-1, api_key=None):
        api_keys = self.api_keys
        api_keys = api_keys.get(name, [])
        if len(api_key) == 0:
            raise ValueError(f"api_key {name} does not exist")
        api_key.pop(index)
        
        self.save_api_keys(api_keys)

    def get_api_keys(self, name):
        return self.api_keys.get(name, [])

    def get_api_key(self, name):
        return c.choice(self.get_api_keys(name))

    def save_api_keys(self, api_keys):
        self.put(self.path, api_keys)
