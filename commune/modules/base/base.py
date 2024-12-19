import commune as c

class Demo:
    def __init__(self, a=1, b=2):
        self.config = c.munch({"a": a, "b": b})

    def generate(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    def test(self, x:int = 1, y:int = 2) -> int:
        return self.generate(x, y)
    