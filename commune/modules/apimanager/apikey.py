

import commune as c 

class ApiManager:
    def __init__(self, module='model.openrouter', path:str=None):
        self.set_module(module)
        self.path = path or c.resolve_path('api')

    def set_module(self, module):
        self.module = module
    def get(self, module=None):
        return c.choice(self.keys(module))
    
    def put(self, key:str, module=None):
        path = self.get_path(module)
        keys = c.get(path, [])
        keys.append(key)
        keys = list(set(keys))
        c.put(path, keys)
        assert key in self.keys(module), f'Error adding api key {key}'
        return {'keys': keys}
    
    def set_keys(self, keys:str , module=None):
        path = self.get_path(module)
        keys = list(set(keys))
        return c.put(path, keys)
    
    def rm(self, module,  key:str):
        module = module or self.module
        keys = self.keys(module)
        n = len(keys)
        if key in keys:
            keys.remove(key)
            self.set_keys(keys, module=module)
        else:
            raise ValueError(f'Error removing api key {key}')
        assert len(self.keys(module)) == n - 1, f'Error removing api key {key}'
        if len(self.keys(module)) == 0:
            self.rm(module)
        return {'keys': keys}

    def get_path(self, module=None):
        module = module or self.module
        return c.resolve_path(self.path + '/' + module + '.json')

    def keys(self, module=None):
        path = self.get_path(module)
        return c.get(path, [])
    
    def api2path(self):
        files = c.files(self.path)
        return {'.'.join(file.split('/')[-1].split('.')[:-1]):file for file in files}
    def apis(self):
        return list(self.api2path().keys())
    def rm(self, module):
        path = self.get_path(module)  
        assert c.path_exists(path), f'Error removing api keys for {module}'
        return c.rm(path)
    def api_paths(self):
        return list(self.api2path().values())
    def api_exists(self, module):
        return module in self.apis()
    def key_exists(self, key, module=None):
        return key in self.keys(module)
    def rm_keys(self, module):
        path =  self.get_path(module)  
        assert c.path_exists(path), f'Error removing api keys for {module}'
        return c.rm(path) 
    