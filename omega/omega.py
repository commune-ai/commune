import commune as c

class Omega(c.Module):
    def __init__(self, a=1):
        self.a = a
    def forward(self, a=1, b = 1):
        return self.a + b + 1


