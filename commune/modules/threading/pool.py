

# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ThreadPoolExecutor."""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'

import os
import sys
import bittensor
from concurrent.futures import _base
import itertools
import queue
import random
import threading
import weakref
import time
from loguru import logger
import commune as c

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


class _WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.start_time = time.time()
        self.args = args
        self.kwargs = kwargs
        

    def run(self):

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as e:
            result = {'error': e}
        self.future.set_result(result)

class ThreadingPool(_base.Executor):
    """ Base threadpool executor with a priority queue 
    """
    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__

    def __init__(self, 
                 maxsize = -1, 
                 max_workers=None, 
                 thread_name_prefix='ThreadRipper'):

        if max_workers is None:
            # Use this number because ThreadPoolExecutor is often
            # used to overlap I/O instead of CPU work.
            max_workers = (os.cpu_count() or 1) * 5

            
        self._max_workers = max_workers
        self.queue = queue.Queue(maxsize = maxsize)
        self.idle_semaphore = threading.Semaphore(0)
        self.threads = {}
        self.broken = False
        self.shutdown = False
        self.shutdown_lock = threading.Lock()
        self.thread_name_prefix = thread_name_prefix

    @property
    def is_empty(self):
        return self.queue.empty()

    def submit(self, module:str, fn:str, args = None, kwargs = None):
        with self.shutdown_lock:
            args = args if args!= None else []
            kwargs = kwargs if kwargs != None else {}
            request = {
                'module': module,
                'fn': fn,
                'args': args,
                'kwargs': kwargs
            }
            self.queue.put(request, block=False)
            self.adjust_thread_count()
            return f

    def adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self.idle_semaphore.acquire(timeout=0):
            return
        num_threads = len(self.threads)
        if num_threads < self._max_workers:
            thread_name = f'{self.thread_name_prefix}-{num_threads}'
            t = threading.Thread(name=thread_name, target=self.worker)
            t.daemon = True
            t.start()
            self.thread.append(t)


    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True
            self.queue.put(NULL_ENTRY)
        
        if wait:
            for t in self.threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass
    shutdown.__doc__ = _base.Executor.shutdown.__doc__




    def worker(self):
        try:
            while True:
                request = self.queue.get(block=True)
                module = request['module']
                fn = request['fn']
                args = request['args']
                kwargs = request['kwargs']
                
                module = c.connect(module)


                if priority == sys.maxsize:
                    del item
                elif item is not None:
                    item.run()
                    # Delete references to object. See issue16284
                    del item
                    continue
                    
                executor = executor_reference()
                # Exit if:
                #   - The interpreter is shutting down OR
                #   - The executor that owns the worker has been collected OR
                #   - The executor that owns the worker has been shutdown.
                if _shutdown or executor is None or executor._shutdown:
                    # Flag the executor as shutting down as early as possible if it
                    # is not gc-ed yet.
                    if executor is not None:
                        executor._shutdown = True
                    # Notice other workers
                    work_queue.put(NULL_ENTRY)
                    return
                del executor
        except BaseException:
            logger.error('work_item', work_item)
            _base.LOGGER.critical('Exception in worker', exc_info=True)

