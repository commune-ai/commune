import commune as c


class StorageVali(c.Module):
    def __init__(self, module = 'storage', tag=None)
        config = self.set_config(kwargs=locals())

        history = {}

        if config.module == None:
            self.storage = c.module(module)




    storage_peers = {}


    def get_storage_peer(self, name, default=None) -> Dict:
        if default == None:
            default = {'address': None,  'size': 0, 'stored_keys': [], 'w': 0}
        return self.get(f'{self.tag}/peers/{name}', {})


    def score_module(self, module, **kwargs) -> float:


        info = module.info()
        obj_keys = self.storage.ls_keys()
        key = c.choice(obj_keys)
        
        obj = self.storage.get(key)

        obj_size = c.sizeof(obj)

        remote_has = self.remote_has(remote_obj_key, module=module)
        if not remote_has:
            module.put(key, obj)
        

        storage_peer = self.get_storage_peer(info['name'])

        remote_has = self.remote_has(remote_obj_key, module=module)
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        
        if not remote_has:
            remote_obj_key = obj['hash']
            module.put(remote_obj_key, obj)
            remote_has = self.remote_has(remote_obj_key, module=module)
        if remote_has:
            if key not in storage_peer['stored_keys']:
                storage_peer['stored_keys'] += [key]
                storage_peer['size'] += obj_size
        else:

            storage_peer['size'] -= obj_size
            storage_peer['size'] = max(storage_peer['size'], 0)
            storage_peer['stored_keys'] = [k for k in storage_peer['stored_keys'] if k != key]

        # set weight
        storage_peer['w'] = storage_peer['size']
        

        return storage_peer
        


    def put(self, *args,**kwargs):
        return self.storage.put(*args,**kwargs)

        
    def get(self, *args,**kwargs):
        return self.storage.get(*args,**kwargs)

