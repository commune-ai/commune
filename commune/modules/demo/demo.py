import commune as c

class Demo(c.Module):
    def __init__(self, config = None, **kwargs):
        self.set_config(config, kwargs=kwargs)
    def bro(self, x='fam'):
        return f'whadup {x}'
    
    def hey(self, x='fam'):
        return f'whadup {x}'
    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)
        print('Testing demo')
        c.print(self.config)
        print(self.bro())
        print(self.hey())
        assert self.bro() == 'whadup fam'
        assert self.hey() == 'whadup fam'
        return True
    