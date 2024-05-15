import commune as c

class Api(c.Module):
    def __init__(self, module=None):
        self.module = c.module(module)
        
    def get_api_key(self, module=None):
        if module != None:
            cls = c.module(module)
        api_keys = cls.api_keys()
        if len(api_keys) == 0:
            return None
        else:
            return c.choice(api_keys)

    def api_keys(self):
        return c.get(self.module.resolve_path('api_keys'), [])

    def rm_api_keys(self):
        c.put(self.resolve_path('api_keys'), [])
        return {'api_keys': []}

    def send_api_keys(self, module:str, network='local'):
        api_keys = self.api_keys()
        assert len(api_keys) > 0, 'no api keys to send'
        module = c.connect(module, network=network)
        return module.add_api_keys(api_keys)
