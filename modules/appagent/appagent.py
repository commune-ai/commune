import commune as c

class Demo(c.Module):
    git_url = "https://github.com/mnotgod96/AppAgent"
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def clone(self):
        return c.cmd(f"git clone {self.git_url}")
        
        return Demo(a=self.config.a, b=self.config.b