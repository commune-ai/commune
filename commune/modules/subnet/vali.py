import commune as c

class Vali(c.Vali):
    def score(self, module):
        x = 1
        return int(module.forward(x) == x)
    