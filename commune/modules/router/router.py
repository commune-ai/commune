import commune as c
import concurrent
class Router(c.Module):

    def __init__(self, max_workers=):
        self.task_map = {}
        self.executor = c.module('executor')()

    


    def submit(self, module : str, fn: str, args=None, kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        c.fleet()


    
    # @classmethod
    # def test(cls):


