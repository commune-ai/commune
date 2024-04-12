import commune as c
from .utils import a

class C(c.Module):
    def __init__(self, a=a):
        self.a = a
    def call(self, b = 1):
        return self.a + b


