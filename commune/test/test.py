       
import commune as c
class Test(c.Module):
    base_modules = ['server', 'key', 'namespace',  'executor', 'vali']

    def test(self,
             *modules,
             module = 'module',
              timeout=40):
        
        if len(modules) == 0:
            modules = self.base_modules

        module2result = {}
        for  module in modules:
            module2result[module] = c.module(module).test()

        return module2result

        
        

        