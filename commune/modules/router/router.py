import commune as c
import concurrent
class Router(c.Module):

    def __init__(self, max_workers=):
        self.task_map = {}
        self.executor = c.module('executor')()

    


    def submit(self, module : str, fn: str, args=None, kwargs=None, timeout=10, priority=1, return_future = False):
        kwargs = {'module': module, 'fn': fn, 'args': args, 'kwargs': kwargs}
        future = self.executor.submit(fn=c.call, kwargs=args, timeout=timeout, priority=priority)
        if return_future:
            return future
        else:
            return future.result()
        
        return future

    def 


    
    @classmethod
    def test(cls):


