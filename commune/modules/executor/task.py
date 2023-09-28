
import commune as c
class Task(c.Module):
    def __init__(self, fn, args, kwargs, timeout:int=10):
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
            time.time() - self.start_time > self.timeout
        ):
            return


        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            result = c.detailed_error(e)

        # set the result of the future
        self.future.set_result(result)

        return result

    def result(self):
        return self.future.result()
    
