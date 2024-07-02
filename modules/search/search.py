import commune as c

class Search(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)

    @classmethod
    def call(self, search = None, tree='main') -> int:
        return c.modules(search)