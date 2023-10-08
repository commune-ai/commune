
import commune as c
from concurrent.futures._base import Future
import time

class Task(c.Module):
    def __init__(self,
                fn:str,
                args:list, 
                kwargs:dict, 
                timeout:int=10, 
                priority:int=1, 
                path: str =None, 
                **extra_attributes
                ):
        
        self.future = Future()
        self.fn = fn # the function to run
        self.start_time = time.time() # the time the task was created
        self.args = args # the arguments of the task
        self.kwargs = kwargs # the arguments of the task
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.path = path # the path to store the state of the task
        self.status = 'pending' # pending, running, done
        self.output # the result of the task

        # for the sake of simplicity, we'll just add all the extra kwargs to the task object
        self.extra_attributes = extra_attributes
        self.__dict__.update(extra_attributes)

        # store the state of the task if a path is given
        if self.path:
            self.save()
    @property
    def lifetime(self) -> float:
        return time.time() - self.start_time

    @property
    def save(self):
        self.put(self.path, self.state())

    def state(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout,
            'start_time': self.start_time, 
            'lifetime': self.lifetime, #
            'status': self.status,
            'output': self.output, 
            **{k: self.__dict__[k] for k in self.extra_attributes.keys()}
        }

    
    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            (time.time() - self.start_time) > self.timeout
        ):
            self.future.set_exception(TimeoutError('Task timed out'))

        # set it to running
        self.status = 'running'

        try:
            # run the function fam
            output = self.fn(*self.args, **self.kwargs)
            # yo dawg, the task finished, so set it to done
            self.status = 'done'
        except Exception as e:
            # what does this do? A: it sets the exception of the future, and sets the status to failed
            output = c.detailed_error(e)
            self.status = 'failed'

        # store the result of the task
        self.output = output      

        # store the state of the task (if path is set)
        if self.path:
            self.save()
        
        # set the result of the future
        self.future.set_result(output)

    def result(self) -> object:
        return self.future.result()

    def cancel(self) -> bool:
        self.future.cancel()

    def running(self) -> bool:
        return self.future.running()
    
    def done(self) -> bool:
        return self.future.done()

    

    
