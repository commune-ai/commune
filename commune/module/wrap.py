
import commune as c
from typing import *

class ModuleWrapper(c.Module):
    protected_attributes = [ 'info', 'serve', 'module_file', 'module_path', 'server_name',  ]
    def __init__(self, 
                 module:'Any' = None, 
                 protected_attributes:List[str] = None,
                  ): 
        if module is None:
            module = c.module()()
        self.module = module
        self.protected_attributes = protected_attributes or self.protected_attributes
        for attr in dir(self.module):
            if attr not in self.whitelist:
                if attr in self.protected_attributes:
                    continue
                if '__' not in attr:
                    try:
                        setattr(self, attr, getattr(self.module, attr))
                    except Exception as e:
                        c.print(f'Error: {e}')
        
    @classmethod
    def module_file(cls): 
        return cls.get_module_path(simple=False)
    
    @classmethod
    def module_path(cls) -> str:
        return module_class.__name__.lower()
