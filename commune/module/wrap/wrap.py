
import commune as c
from typing import *

class ModuleWrapper(c.Module):
    protected_attributes = [ 'info', 'serve', 'module_file', 'module_path', 'server_name',  'test']
    
    def __init__(self, 
                 module:'Any' = None
                  ): 
        self.module = module
        
    @classmethod
    def module_file(cls): 
        return cls.get_module_path(simple=False)
    
    def module_path(cls) -> str:
        return cls.__name__.lower()


    def schema(self, **kwargs) -> Dict[str, Any]:
        return c.schema(module=self.module, **kwargs)
    
    
    def functions(self, ) -> Dict[str, Any]:
        return c.get_functions(module=self.module)
    

        
    def __getattr__(self, key):

        if key in self.protected_attributes :
            return getattr(self, key)
        else:
            return lambda *args, **kwargs : getattr(self.module, (key))( *args, **kwargs)

