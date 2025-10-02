
import os
import sys
import time
import queue
import weakref
import itertools
import threading
import asyncio
from loguru import logger
from typing import *
from concurrent.futures._base import Future
import time
from tqdm import tqdm
from .utils import detailed_error

class Task:
    def __init__(self, 
                fn:Union[str, callable],
                params:dict, 
                timeout:int=10, 
                priority:int=1, 
                path = None, 
                **extra_kwargs):
        
        self.fn = fn if callable(fn) else lambda *args, **kwargs: fn
        self.set_params(params)
        self.start_time = time.time() # the time the task was created
        self.end_time = 0
        self.timeout = timeout # the timeout of the task
        self.priority = priority # the priority of the task
        self.data = None # the result of the task
        self.path = os.path.abspath(path) if path != None else None
        self.status = 'pending' # pending, running, done
        self.future = Future()

    def set_params(self, params):
        self.params = params
        if 'args' in self.params and 'kwargs' in self.params:
            self.params['args'], self.params['kwargs'] = self.params.get('args', []), self.params.get('kwargs', {})
        else:
            if isinstance(params, dict) and len(params) > 0:
                self.params = dict(args=[], kwargs=params)
            elif isinstance(params, list) and len(params) > 0:
                self.params = dict(args=params, kwargs={})
        return self.params

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def state(self) -> dict:
        return {
            'fn': self.fn.__name__,
            'params': self.params,
            'timeout': self.timeout,
            'start_time': self.start_time, 
            'end_time': self.end_time,
            'priority': self.priority,
            'status': self.status,
        }

    def run(self):
        """Run the given work item"""
        # Checks if future is canceled or if work item is stale

        if (not self.future.set_running_or_notify_cancel()) or (time.time() - self.start_time) > self.timeout:
            self.future.set_exception(TimeoutError('Task timed out'))
        try:
            data = self.fn(*self.params['args'], **self.params['kwargs'])
            self.status = 'complete'
        except Exception as e:
            data = detailed_error(e)
            self.status = 'failed'
        self.future.set_result(data)
        self.data = data 
        self.end_time = time.time()
              
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


NULL_TASK = (sys.maxsize, Task(None, {}))
