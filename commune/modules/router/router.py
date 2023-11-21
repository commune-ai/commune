import commune as c
Thread = c.module('thread')
import asyncio
import gc
class Router(c.Module):
    default_fn = 'info'
    fn_splitter = '/'

    def __init__(self, max_workers=10):
        self.executor = c.module('executor.thread')(max_workers=max_workers)


    def call(self, server, *args, network:str='local', **kwargs):
        if fn_splitter in server:
            server, fn = module.split(fn_splitter)
        else:
            fn = default_fn
        
        output  = c.call(server, fn, *args, **kwargs)
        self.executor.submit(c.call, server, fn, args=args, kwargs=kwargs)


        return output


    def servers(self, network:str='local'):
        return c.servers(network=network)
    
    def namespace(self, network:str='local'):
        return c.namespace(network=network)



                
        



        


