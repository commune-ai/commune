import commune as c


class Template(c.Module):
    def __init__(self, obj=None, ):
        is_fn = callable(obj)
        if is_fn:
            self.set_fn(obj)
        else:
            self.set_module(obj)


    def set_module(self, module):
        if module == None:
            return None
        self.module = module
        for fn in dir(module):
            setattr(self, fn, getattr(self, fn)) 

    def set_fn(self, fn):
        if fn == None:
            return None
        self.fn = fn


    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
                                   
    def from_fn(cls, fn):
        return cls(fn=fn)
    
    def from_module(cls, module):
        return cls(module=module)
    


    def test(self):
        print('test')
        template = c.module('template')()

        def fn1():
            return 0
        
        def fn2():
            return 1
        
        template.set_fn(fn1)
        assert template.forward() == 0
        template.set_fn(fn2)
        assert template.forward() == 1

        class Custom(c.Module):

            def test(self, a=1, b=2):
                return a+b

        custom = Custom()
        template = c.module('template')(obj=custom)
        assert template.forward(a=3, b=4) == 7
        print('template test passed')
    

    
