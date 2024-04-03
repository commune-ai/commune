import commune as c
class Vali(c.m('vali')):
    subnet = 'add'
    network = 'local'

    def resolve_module_client(self, module):
        if isinstance(module, str):
            module = c.connect(module, network=self.network, key=self.key)
        return module

    
    def score_module(self, module):
        module  = self.resolve_module_client(module)
        result = module.add(1, 1)
        assert result == 2, f"result: {result}"
        return 1
        
