import commune as c

class Custom(c.Module):
    def __init__(self, module=None, **kwargs):
        self.set_config(kwargs=locals())

    def add_fn(self, fn):
        assert callable(fn)
        setattr(self, fn.__name__, fn)

    def fn_exists(self, fn:str = 'info'):
        return hasattr(self, fn)
    
    def whitelist_fn(self, fn):
        assert self.fn_exists(fn)
        self.whitelist.append(fn)
        return {fn: 'whitelisted', 'whitelist': self.whitelist}

    def blacklist_fn(self, fn):
        assert self.fn_exists(fn)
        self.blacklist.append(fn)
        return {fn: 'blacklisted', 'blacklist': self.blacklist}


    def rm_fn(self, fn):
        assert self.fn_exists(fn)
        delattr(self, fn)
        assert not self.fn_exists(fn)
        return {'success': True, 'fn': fn, 'msg': 'removed'}
    
    
    def add_fns(self, *fns):
        if len(fns) == 1 and isinstance(fns[0], (list, tuple)):
            fns = fns[0]
        for fn in fns:
            self.add_fn(fn)

    def add_module(self, module):
        assert isinstance(module, c.Module)
        for fn in module.fns():
            self.add_fn(fn)

    def add_modules(self, *modules):
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = modules[0]
        for module in modules:
            self.add_module(module)


    def test(self):
        print('test')
        custom = c.module('custom')()

        def fn1():
            return 0
        
        def fn2():
            return 1
        

        custom.add_fns(fn1, fn2)

        assert custom.fn1() == 0
        assert custom.fn2() == 1
        return True


    


    