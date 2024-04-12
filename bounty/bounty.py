import commune as c

class Bounty(c.Module):
    def __init__(self, a=1):
        self.a = a
    def forward(self, b = 1):
        return self.a + b


