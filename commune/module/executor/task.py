# Workers are created as daemon threads. This is done to allow the interpreter
# to exit when there are still idle threads in a ThreadPoolExecutor's thread
# pool (i.e. shutdown() was not called). However, allowing workers to die with
# the interpreter has two undesirable properties:
#   - The workers would still be running during interpreter shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads finish.

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
                save:bool = False,
                path = None,
                **extra_kwargs):
        
        self.future = Future()
        self.fn = fn # the function to run
        self.start_time = time.time() # the time the task was created
        self.args = args # the arguments of the task
        self.kwargs = kwargs # the arguments of the task
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.data = None # the result of the task
    
        self.fn_name = fn.__name__ if fn != None else str(fn) # the name of the function
        # for the sake of simplicity, we'll just add all the extra kwargs to the task object
        self.extra_kwargs = extra_kwargs
        self.save = save
        self.status = 'pending' # pending, running, done
        self.__dict__.update(extra_kwargs)
        # save the task state


    @property
    def lifetime(self) -> float:
        return time.time() - self.start_time

    @property
    def state(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout,
            'start_time': self.start_time, 
            'priority': self.priority,
            'status': self.status,
            'data': self.data, 
            **self.extra_kwargs
        }
    
    @property
    def save_state(self):
        
        self.path
        path = f"{self.status}_{self.fn_name}_args={str(self.args)}_kwargs={str(self.kwargs)}"
        if self.path != None:
            path = f"{self.path}/{path}"
        if self.status == 'pending':
            return self.put(self.status2path[self.status], self.state)
        elif self.status in ['complete', 'failed']:
            if c.exists(self.paths['pending']):
                c.rm(self.paths['pending'])
            return self.put(self.paths[self.status], self.state)
        else:
            raise ValueError(f"Task status must be pending or complete, not {self.status}")
    
    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            (time.time() - self.start_time) > self.timeout
        ):
            self.future.set_exception(TimeoutError('Task timed out'))

        try:
            data = self.fn(*self.args, **self.kwargs)
            self.status = 'complete'
        except Exception as e:

            # what does this do? A: it sets the exception of the future, and sets the status to failed
            data = c.detailed_error(e)
            if 'event loop' in data['error']: 
                c.new_event_loop(nest_asyncio=True)
            self.status = 'failed'

        self.future.set_result(data)
        # store the result of the task
        
        self.data = data       

        if self.save:
            self.save_state()

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



