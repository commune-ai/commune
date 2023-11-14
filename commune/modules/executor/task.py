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


import os
import sys
import time
import queue
import random
import weakref
import itertools
import threading

from loguru import logger
from typing import Callable
import concurrent
from concurrent.futures._base import Future
import commune as c
import gc




class Task(c.Module):
    def __init__(self, fn:str, args:list, kwargs:dict, timeout:int=10, priority:int=1, path=None, **extra_kwargs):
        self.future = Future()
        self.fn = fn # the function to run
        self.start_time = time.time() # the time the task was created
        self.args = args # the arguments of the task
        self.kwargs = kwargs # the arguments of the task
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.path = path # the path to store the state of the task
        self.status = 'pending' # pending, running, done
        self.data = None # the result of the task

        # for the sake of simplicity, we'll just add all the extra kwargs to the task object
        self.extra_kwargs = extra_kwargs
        self.__dict__.update(extra_kwargs)

        # store the state of the task if a path is given
        if self.path:
            self.save()
    @property
    def lifetime(self) -> float:
        return time.time() - self.start_time

    @property
    def save(self):
        self.put(self.path, self.state)

    @property
    def state(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout,
            'start_time': self.start_time, 
            'priority': self.lifetime,
            'status': self.status,
            'data': self.data, 
            **{k: self.__dict__[k] for k,v in self.extra_kwargs.items()}
        }

    

    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (
            (time.time() - self.start_time) > self.timeout
        ):
            self.future.set_exception(TimeoutError('Task timed out'))

        self.status = 'running'
        try:
            c.print(f'Calling {self.fn}', self.kwargs)
            data = self.fn(*self.args, **self.kwargs)
            
            self.status = 'done'
        except Exception as e:
            # what does this do? A: it sets the exception of the future, and sets the status to failed
            self.status = 'failed'
            data = c.detailed_error(e)
        
        self.future.set_result(data)
        # store the result of the task
        self.data = data       

        # store the state of the task 
        if self.path:
            self.save()

        # set the result of the future
        

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
    

