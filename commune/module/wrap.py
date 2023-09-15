
import commune as c

class ModuleWrapper(c.Module):
    def __init__(self, 
                 module:'Any', 
                 whitelist = None
                  ): 
        if whitelist == None:
            module_fns = c.get_functions(module)
        
        c.__init__(self, *args, **kwargs) 
        self.merge(self.module)
        
    @classmethod
    def module_file(cls): 
        return cls.get_module_path(simple=False)
    
    
    def __call__(self, *args, **kwargs):
        return self.module.__call__(self, *args, **kwargs)

    def __str__(self):
        return self.module.__str__()
    
    def __repr__(self):
        return self.module.__repr__() 
    @classmethod
    def module_path(cls) -> str:
        return module_class.__name__.lower()

    @classmethod
    def functions(cls):
        return cls.get_functions(module)


if is_class:
    return ModuleWrapper
else:
    return ModuleWrapper(module=module)