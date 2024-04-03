import commune as c
class Vali(c.m('vali')):
    def __init__(self, 
                 network='local', 
                 search = 'subnet.add.miner',
                 netuid = 1,
                   **kwargs):
        self.init(locals())

    def score_module(self, module):
        if isinstance(module, str): 
            module = c.connect(module)

        result = module.add(1, 1)
        assert result == 2, f"result: {result}"
        return 1
        
