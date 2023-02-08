
import commune
import os


class DemoModule:
    def __init__(self,x='bro'):
        self.x = x  
    def return_x(self, x = None):
        x = x if x else self.x
        return x

def test_module_inheritance(x='bro'):
    
    self = commune.module(DemoModule)(x=x )
    assert self.return_x(x) == x
    assert self.module_name() == 'DemoModule'
    
    self = commune.module(DemoModule(x=x))
    assert self.return_x(x) == x
    
    
    assert self.module_name() == 'DemoModule'

    
def test_serve(x='bro'):
    self = commune.module(DemoModule)(x=x )
    self.serve(wait_for_termination=False, verbose=False)
    
    print(self.server_stats)
    client_module = self.connect('DemoModule')

    
    client_module.return_x(x) == 'x'
    client_module.return_x(x=x) == 'x'
    
    assert 'DemoModule' in commune.servers()
    print(self.kill_server(self.server_stats['port']))
    assert 'DemoModule' not in commune.servers(), commune.servers()
    

def test_load( config_path = './commune/module.yaml'):
    
    import munch
    
    for config in [False, config_path, None ]:
        module = commune.Module()
        assert isinstance(module, commune.Module)
        assert hasattr(module, 'config')
        assert isinstance(module.config, munch.Munch)
    
    



if __name__ == '__main__':
    test_serve()
    test_module_inheritance()
    test_load()