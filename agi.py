import commune as c

class Agi:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def generate(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    
    forward = generate