import commune as c

Vali = c.module("vali")
class Vali(Vali):
    def __init__(self, 
                 network='local', # the network is local
                 search = 'miner',
                 netuid = 1,
                   **kwargs):
        # send locals() to init
        self.init(locals())

    def score_module(self, module):
        if isinstance(module, str): 
            module = c.connect(module)
        result = module.forward(1, 1)
        assert result == 2, f"result: {result}"
        return 1
    
    def forward(self, x, y):
        return x + y
    
    