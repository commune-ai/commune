import commune as c

class Router(c.Module):

    def __init__(self,  **kwargs):
        self.executor = c.m('executor')(**kwargs)


    def submit(self, module: str = 'module/info', 
               *args,
               params : dict = None, 
               timeout : int =4 
               ):
        if len(args) == 1:
            params = args[0]
        module, fn = module.split('/')
        module = c.module(module)
        fn = getattr(module, fn)
        assert callable(fn)
        self.executor.submit(fn, params=params timeout=timeout)
        



    