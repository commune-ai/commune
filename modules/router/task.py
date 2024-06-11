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

class RouterTask(c.Module):
    def __init__(self, 
                 module:str = 'module',
                fn:str = 'info',
                args:list  = None, 
                kwargs:dict = None, 
                timeout:int=10, 
                priority:int=1, 
                path=None,
                save:bool = True,
                fn_seperator:str = '/',
                network : str = 'local',
                namespace = None,
                **extra_kwargs):
        if fn_seperator in str(module):
            module, fn = module.split(fn_seperator)
            
        self.fn_seperator = fn_seperator
        self.network = network
        self.module = module
        self.fn = fn # the name of the function
        self.path = path
        self.start_time = int(time.time()) # the time the task was created
        self.args = args if args != None else [] # the arguments of the task
        self.kwargs = kwargs if kwargs != None else {} # the arguments of the task
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.path = path # the path to store the state of the task
        self.data = None # the result of the task
        self.save = save
        self.extra_kwargs = extra_kwargs
        self.__dict__.update(extra_kwargs)
        # save the task state
        self.status = 'pending' # pending, running, done
        self.future = Future()


    @property
    def lifetime(self) -> float:
        return time.time() - self.start_time

    @property
    def state(self) -> dict:
        return {
            'network': self.network,
            'module': self.module,
            'fn': self.fn,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout,
            ## other stuff
            'start_time': self.start_time, 
            'priority': self.lifetime,
            'status': self.status,
            # return the data if the task is complete
            'data': self.data, 
            **self.extra_kwargs
        }
    

    def save_state(self):
        self.status2path = {status: f'{status}/module={self.module}_fn={self.fn}_ts={self.start_time}' for status in ['pending', 'complete', 'failed']}
        for k,v in self.status2path.items():
            if self.path != None:
                self.status2path[k] = f"{self.path}/{v}"
        path = self.status2path[self.status]
        if self.status == 'pending':
            return self.put(path, self.state)
        elif self.status in ['complete', 'failed']:
            if c.exists(self.status2path['pending']):
                c.rm(self.status2path['pending'])
            return self.put(path, self.state)
        else:
            raise ValueError(f"RouterTask status must be pending or complete, not {self.status}")
    
    def run(self):
        # SAVE AT THE PENDING TRANSACTION
        self.status = 'pending'
        if self.save:
            self.save_state()
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            (time.time() - self.start_time) > self.timeout
        ):
            self.future.set_exception(TimeoutError('RouterTask timed out'))
        try:

            # connect to module
            module = c.connect(self.module, network=self.network)
            
            # run the function
            data = getattr(module, self.fn)(*self.args, timeout=self.timeout, **self.kwargs)

            # set the status to complete if the function runs successfully
            self.status = 'complete'
        except Exception as e:

            # what does this do? A: it sets the exception of the future, and sets the status to failed
            data = c.detailed_error(e)
            self.status = 'failed'
        self.data = data       

        if self.save:
            self.save_state()

        self.future.set_result(data)
        # store the result of the task
        

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
        if isinstance(other, RouterTask):
            return self.priority < other.priority
        elif isinstance(other, int):
            return self.priority < other
        else:
            raise TypeError(f"Cannot compare RouterTask with {type(other)}")
    
    @classmethod
    def null(cls):
        return cls()
