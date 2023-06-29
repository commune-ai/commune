import commune as c

class Demo(c.Module):
    def __init__(self, i_hate_configs:bool=True):
        self.i_hate_configs = i_hate_configs
    
    def bro(self, x='fam'):
        return f'whadup {x} i_hate_configs: {self.i_hate_configs}'
    
    def hey(self, x='fam'):
        return f'whadup {x}'
    