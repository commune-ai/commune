import commune as c

class Subnet(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(kwargs=locals())
    
    def forward(self, a=1, b=1):
        return a + b

    def validate(self, module):
        a = 1
        b = 1
        result = module.forward(a, b)
        assert result == (a + b), f"result: {result}"



    def test(self, n=3):
        for i in range(n):
            c.print(c.serve(f'subnet.miner::{i}'))
        c.print(c.serve('subnet.vali', kwargs={'network': 'local'}))
    
