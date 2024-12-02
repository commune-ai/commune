import commune as c


class Model(c.Module):
    def __init__(self, module='chat', network='local'):
        self.module = module
        self.network = network

    def generate(self, *args, **kwargs):
        models = c.servers(self.module)
        client = c.choice(models)
        client = c.connect(client)
        return client.generate(*args, **kwargs)

    
    

