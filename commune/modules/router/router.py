import commune as c
import concurrent
class Router(c.Module):

    def __init__(self, max_workers=):
        self.task_map = {}
        self.executor = c.module('executor')()

    
    def submit(self, module : str, fn: str, args=None, kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        module = c.call(module=module, fn=fn, *args, **kwargs)
        task =  self.executor.submit(module, *args, **kwargs)
        self.task_map[task.] = module