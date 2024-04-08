import commune as c
class Miner(c.Module):
    description = 'This is the miner module that adds two numbers together'
    whitelist = ['forward'] 
    def forward(self, a=1, b=1):
        return a + b
    


        
