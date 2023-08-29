import commune as c

class Access(c.Module):
    def __init__(self, **kwargs):
        config = self.set_config(config=kwargs)
    def verify(self):
        print('Base run')


