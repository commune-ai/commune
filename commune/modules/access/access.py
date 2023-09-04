import commune as c

class Access(c.Module):
    def __init__(self, module, **kwargs):
        config = self.set_config(config=kwargs)
        self.module = module


    

    def verify(self, **kwargs):
        



