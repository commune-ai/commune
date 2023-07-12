
import commune
import os


class DemoModule:
    def __init__(self,x='bro'):
        self.x = x  
    def return_x(self, x = None):
        x = x if x else self.x
        return x

    
def test_serve(x='bro'):
    self = commune.module(DemoModule)(x=x )
    self.serve(wait_for_termination=False, verbose=False)
    
    print(self.server_stats)
    client_module = self.connect('DemoModule')

    
    client_module.return_x(x) == 'x'
    client_module.return_x(x=x) == 'x'
    
    assert 'DemoModule' in commune.servers()
    print(self.kill_port(self.server_stats['port']))
    assert 'DemoModule' not in commune.servers(), f'{commune.servers()}'
    


if __name__ == '__main__':
    test_serve()