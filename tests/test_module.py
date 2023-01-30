
import commune
import os

PWD = os.getenv('PWD')

def test_load(self, config_path = './commune/module.yaml'):
    
    import munch
    
    for config in [False, config_path, None ]:
            module = commune.Module()
            assert isinstance(module, commune.Module)
            
            assert hasattr(module, 'config')
            assert isinstance(module.config, munch.Munch)