import commune as c

class Agi:
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b

    def generate(self, x:int = 1, y:int = 2) -> int:
        c.print(self.a, 'This is a')
        c.print(self.b, 'This is b')
        return x + y
    
    forward = generate