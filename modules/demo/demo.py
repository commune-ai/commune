import commune as c

class Demo(c.Module):
    def __init__(self, a=1):
        self.a = a
    def call(self, b = 1):
        return self.a + b


