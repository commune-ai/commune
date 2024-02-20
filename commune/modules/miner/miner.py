import commune as c

class Demo(c.Module):
    def __init__(self, config= None, **kwargs):
        self.set_config(config, kwargs=kwargs) # This is a Munch object


    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    