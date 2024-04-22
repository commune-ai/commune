import commune as c

class Vali(c.Module):
    def forward(self, x, y):
        return x + y
    def score(self, module, x=1, y=2) -> int:
        z = module.forward(x=x, y=y)
        if z == (x + y):
            return 1
        else:
            return 0
        
    