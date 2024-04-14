import commune as c

class ValiAdd(c.module('vali')):
    def __init__(self, network='local', **kwargs):
        self.init_vali(locals())

    def score_module(self, module) -> int:
        a = c.random_int()
        b = c.random_int()
        result = module.forward(a, b)
        assert result == a + b , f'{result} != {a} + {b}'
        return 1
    
    def forward(self, a=1, b=1):
        return a + b
    