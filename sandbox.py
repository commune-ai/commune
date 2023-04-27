import commune

class Sand(commune.Module):
    def __init__(self):
        self._dynamic_attrs = {}
        
    def __getattr__(self, name):
        if name in self._dynamic_attrs:
            return self._dynamic_attrs[name]
        else:
            return None
        
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._dynamic_attrs[name] = value
            
    def __delattr__(self, name):
        if name in self._dynamic_attrs:
            del self._dynamic_attrs[name]
        else:
            super().__delattr__(name)

s = Sand()
s.dynamic_attr = 42
print(s.dynamic_attr) # prints 42
del s.dynamic_attr
print(s.dynamic_attr) # prints None
