import commune as c

class Vali(c.module("vali")):
    def __init__(self, 
                 network='local', # the network is local
                 search = None,
                   **kwargs):
        # send locals() to init
        self.init_vali(locals())

    def score_module(self, module):
        a = 1
        b = 1
        result = module.forward(a, b)
        assert result == (a+b), f"result: {result}"
        return 1
    

    def forward(self, x, y):
        return x + y
    
    