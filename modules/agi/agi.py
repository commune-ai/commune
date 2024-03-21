import commune as c

class Agi(c.Module):
    def __init__(self, config={'a': 1, 'b': 2}, **kwargs):
        self.set_config(config)
        c.print(self.config, 'This is the config, it is a Munch object')

    def call(self, x:int = 1, y:int = 2) -> int:
        return x + y
    
    def call2(self, x:int = 1, y:int = 2) -> int:
        c.print('fammmm')
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    