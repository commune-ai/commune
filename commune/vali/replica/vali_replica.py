import commune as c
Vali = c.module('vali')
class ValiReplica(Vali):
    def __init__(self, **kwargs):
        config = self.set_config(kwargs=kwargs)
        self.reference_module  = c.module(config.module)()
        Val.init_vali(self, **kwargs)

        
    def score(self, module):
        pass

