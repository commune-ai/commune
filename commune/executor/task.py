import time
from concurrent.futures._base import Future
import commune as c

class Task(c.Module):
    def __init__(self, 
                fn:str,
                args:list, 
                kwargs:dict, 
                timeout:int=10, 
                priority:int=1, 
                path = None, 
                **extra_kwargs):
        
        self.future = Future()
        self.fn = fn # the function to run
        self.start_time = time.time() # the time the task was created
        self.end_time = None
        self.args = args # the arguments of the task
        self.kwargs = kwargs # the arguments of the task
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.data = None # the result of the task
        self.latency = None
    
        self.fn_name = fn.__name__ if fn != None else str(fn) # the name of the function
        # for the sake of simplicity, we'll just add all the extra kwargs to the task object
        self.path = self.resolve_path(path) if path != None else None
        self.status = 'pending' # pending, running, done

    @property
    def lifetime(self) -> float:
        return time.time() - self.start_time

    @property
    def state(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'args': self.args,
            'kwargs': self.kwargs,
            'timeout': self.timeout,
            'start_time': self.start_time, 
            'end_time': self.end_time,
            'latency': self.latency,
            'priority': self.priority,
            'status': self.status,
            'data': self.data, 
        }
    
    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        self.start_time = c.time()

        if (not self.future.set_running_or_notify_cancel()) or (time.time() - self.start_time) > self.timeout:
            self.future.set_exception(TimeoutError('Task timed out'))
        try:
            data = self.fn(*self.args, **self.kwargs)
            self.status = 'complete'
        except Exception as e:
            data = c.detailed_error(e)
            if 'event loop' in data['error']: 
                c.new_event_loop(nest_asyncio=True)
            self.status = 'failed'

        self.future.set_result(data)
        # store the result of the task
        if self.path != None:
            self.save(self.path, self.state)
        
        self.end_time = c.time()
        self.latency = self.end_time - self.start_time
        self.data = data       

    def result(self) -> object:
        return self.future.result()

    @property
    def _condition(self) -> bool:
        return self.future._condition
    @property
    def _state(self, *args, **kwargs) -> bool:
        return self.future._state

    @property
    def _waiters(self) -> bool:
        return self.future._waiters

    def cancel(self) -> bool:
        self.future.cancel()

    def running(self) -> bool:
        return self.future.running()
    
    def done(self) -> bool:
        return self.future.done()

    def __lt__(self, other):
        if isinstance(other, Task):
            return self.priority < other.priority
        elif isinstance(other, int):
            return self.priority < other
        else:
            raise TypeError(f"Cannot compare Task with {type(other)}")
