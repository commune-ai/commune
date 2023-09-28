
import commune as c
from concurrent.futures._base import Future
import time
import gc

class Task(c.Module):
    def __init__(self, fn:str, args:list, kwargs:dict, timeout:int=10):
        self.future = Future()
        self.fn = fn
        self.start_time = time.time()
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

    @property
    def info(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout,
            'start_time': self.start_time
        }

    

    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            (time.time() - self.start_time) > self.timeout
        ):
            self.future.set_exception(TimeoutError('Task timed out'))


        try:
            result = self.fn(*self.args, **self.kwargs)

        except Exception as e:
            result = c.detailed_error(e)
            self.future.set_exception(e)
        # set the result of the future
        self.future.set_result(result)
        del result
        del self.fn
        del self.args
        del self.kwargs

    def result(self):
        return self.future.result()
    
