import commune as c


class StorageVali(c.Module):
    def __init__(self, module = 'storage', tag=None)
        config = self.set_config(kwargs=locals())

        history = {}

        if config.module == None:
            self.storage = c.module(module)




    def score_module(module) -> float:
        remote_has = self.storage.score_module(module)



    def put(self, *args,**kwargs):
        return self.storage.put(*args,**kwargs)

        
    def get(self, *args,**kwargs):
        return self.storage.get(*args,**kwargs)

