import commune as c
class Miner(c.Module):
    whitelist = ['add'] 
    def forward(self, a=1, b=1):
        return a + b
        
