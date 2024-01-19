import commune as c

class Demo(c.Module):
    def __init__(self, a=1):
        self.a = a
    def call_demo(self):
        return c.module('demo')().call()

    def call2222(self, x:int = 1, y:int = 2) -> int:
        c.print(self.a)
        return x + y
    