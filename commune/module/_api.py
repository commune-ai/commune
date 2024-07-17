import os

class Api:
    
    def set_api_key(self, api_key:str, cache:bool = True):
        api_key = os.getenv(str(api_key), None)
        if api_key == None:
            api_key = self.get_api_key()
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)
        assert isinstance(api_key, str)

    
    def add_api_key(self, api_key:str):
        assert isinstance(api_key, str)
        api_keys = self.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        self.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    
    def add_api_keys(self, *api_keys:str):
        if len(api_keys) == 1 and isinstance(api_keys[0], list):
            api_keys = api_keys[0]
        api_keys = list(set(api_keys + self.get('api_keys', [])))
        self.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    
    def set_api_keys(self, api_keys:str):
        api_keys = list(set(api_keys))
        self.put('api_keys', api_keys)
        return {'api_keys': api_keys}

    
    def rm_api_key(self, api_key:str):
        assert isinstance(api_key, str)
        api_keys = self.get(self.resolve_path('api_keys'), [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   
        path = self.resolve_path('api_keys')
        self.put(path, api_keys)
        return {'api_keys': api_keys}


    def get_api_key(self, module=None):
        if module != None:
            self = self.module(module)
        api_keys = self.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return self.choice(api_keys)

    def api_keys(self):
        return self.get(self.resolve_path('api_keys'), [])
    

    def rm_api_keys(self):
        self.put(self.resolve_path('api_keys'), [])
        return {'api_keys': []}

