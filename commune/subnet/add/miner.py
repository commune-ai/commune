import commune as c


class Miner(c.Module):
    def forward(self, a=1, b=2):
        return a + b
    
    