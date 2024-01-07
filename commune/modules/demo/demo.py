import commune as c
#hey /n/n
    def __init__(self, a=1):
        self.a = a

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.a)
        
        return x + y
    