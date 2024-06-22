import commune as c
import os

class Api(c.Module):
    
    def set_api_key(self, api_key:str, cache:bool = True):
        api_key = os.getenv(str(api_key), None)
        if api_key == None:
            api_key = self.get_api_key()

        
        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)


    @classmethod
    def add_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = cls.get('api_keys', [])
        api_keys.append(api_key)
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}


    @classmethod
    def add_api_keys(cls, *api_keys:str):
        if len(api_keys) == 1 and isinstance(api_keys[0], list):
            api_keys = api_keys[0]
        api_keys = list(set(api_keys + cls.get('api_keys', [])))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}
    
    @classmethod
    def set_api_keys(cls, api_keys:str):
        api_keys = list(set(api_keys))
        cls.put('api_keys', api_keys)
        return {'api_keys': api_keys}

    @classmethod
    def rm_api_key(cls, api_key:str):
        assert isinstance(api_key, str)
        api_keys = c.get(cls.resolve_path('api_keys'), [])
        for i in range(len(api_keys)):
            if api_key == api_keys[i]:
                api_keys.pop(i)
                break   
        path = cls.resolve_path('api_keys')
        c.put(path, api_keys)
        return {'api_keys': api_keys}

    @classmethod
    def get_api_key(cls, module=None):
        if module != None:
            cls = c.module(module)
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    @classmethod
    def api_keys(cls):
        return c.get(cls.resolve_path('api_keys'), [])
    

    @classmethod
    def rm_api_keys(self):
        c.put(self.resolve_path('api_keys'), [])
        return {'api_keys': []}


    ## API MANAGEMENT ##

    @classmethod
    def send_api_keys(cls, module:str, network='local'):
        api_keys = cls.api_keys()
        assert len(api_keys) > 0, 'no api keys to send'
        module = c.connect(module, network=network)
        return module.add_api_keys(api_keys)