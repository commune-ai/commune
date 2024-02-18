import commune as c

class Demo(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def create_dope_stuff(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    