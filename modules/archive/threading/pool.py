""" Factory method for creating priority threadpool
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import copy
import bittensor
from . import priority_thread_pool_impl

class ThreadRipper:
    """ Factory method for creating priority threadpool
    """
    def __new__(
            cls,
            config: 'bittensor.config' = None,
            max_workers: int = None,
            maxsize: int = None,
        ):
        r""" Initializes a priority thread pool.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.subtensor.config()
                max_workers (default=10, type=int)
.                   The maximum number of threads in thread pool
                maxsize (default=-1, type=int)
                    The maximum number of tasks in the priority queue
        """        
        if config == None: 
            config = prioritythreadpool.config()
        config = copy.deepcopy( config )
        config.axon.priority.max_workers = max_workers if max_workers != None else config.axon.priority.max_workers
        config.axon.priority.maxsize = maxsize if maxsize != None else config.axon.priority.maxsize

        prioritythreadpool.check_config( config )
        return priority_thread_pool_impl.PriorityThreadPoolExecutor(maxsize = config.axon.priority.maxsize, max_workers = config.axon.priority.max_workers)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'axon.priority.max_workers', type = int, help='''maximum number of threads in thread pool''', default = bittensor.defaults.axon.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.priority.maxsize', type=int, help='''maximum size of tasks in priority queue''', default = bittensor.defaults.axon.priority.maxsize)  
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.axon = bittensor.Config()
        defaults.axon.priority = bittensor.Config()
        defaults.axon.priority.max_workers = os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') if os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') != None else 5
        defaults.axon.priority.maxsize = os.getenv('BT_AXON_PRIORITY_MAXSIZE') if os.getenv('BT_AXON_PRIORITY_MAXSIZE') != None else 10
    
    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
            Return: bittensor.config object 
        """
        parser = argparse.ArgumentParser()
        prioritythreadpool.add_args( parser )
        return bittensor.config( parser )
    
    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for threadpool worker number and size
        """
        assert isinstance(config.axon.priority.max_workers, int), 'axon.priority.max_workers must be a int'
        assert isinstance(config.axon.priority.maxsize, int), 'axon.priority.maxsize must be a int'




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

_threads_queues = weakref.WeakKeyDictionary()
_shutdown = False

class _WorkItem(object):
    def __init__(self, future, fn, start_time, args, kwargs):
        self.future = future
        self.fn = fn
        self.start_time = start_time
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """ Run the given work item
        """
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (time.time()-self.start_time > bittensor.__blocktime__):
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)


NULL_ENTRY = (sys.maxsize, _WorkItem(None, None, time.time(), (), {}))

def _worker(executor_reference, work_queue, initializer, initargs):
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            executor = executor_reference()
            if executor is not None:
                executor._initializer_failed()
            return
    try:
        while True:
            work_item = work_queue.get(block=True)
            priority = work_item[0]
            item = work_item[1]
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


class BrokenThreadPool(_base.BrokenExecutor):
    """
    Raised when a worker thread in a ThreadPoolExecutor failed initializing.
    """


class PriorityThreadPoolExecutor(_base.Executor):
    """ Base threadpool executor with a priority queue 
    """
    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__

    def __init__(self, maxsize = -1, max_workers=None, thread_name_prefix='',
                 initializer=None, initargs=()):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
            initializer: An callable used to initialize worker threads.
            initargs: A tuple of arguments to pass to the initializer.
        """
        if max_workers is None:
            # Use this number because ThreadPoolExecutor is often
            # used to overlap I/O instead of CPU work.
            max_workers = (os.cpu_count() or 1) * 5
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")

        self._max_workers = max_workers
        self._work_queue = queue.PriorityQueue(maxsize = maxsize)
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = set()
        self._broken = False
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread_name_prefix = (thread_name_prefix or
                                    ("ThreadPoolExecutor-%d" % self._counter()))
        self._initializer = initializer
        self._initargs = initargs

    @property
    def is_empty(self):
        return self._work_queue.empty()

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise BrokenThreadPool(self._broken)

            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            if _shutdown:
                raise RuntimeError('cannot schedule new futures after '
                                   'interpreter shutdown')

            priority = kwargs.get('priority', random.randint(0, 1000000))
            if priority == 0:
                priority = random.randint(1, 100)
            eplison = random.uniform(0,0.01) * priority
            start_time = time.time()
            if 'priority' in kwargs:
                del kwargs['priority']
            

            f = _base.Future()
            w = _WorkItem(f, fn, start_time, args, kwargs)
            self._work_queue.put((-float(priority + eplison), w), block=False)
            self._adjust_thread_count()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__


    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(NULL_ENTRY)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=_worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs))
            t.daemon = True
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue

    def _initializer_failed(self):
        with self._shutdown_lock:
            self._broken = ('A thread initializer failed, the thread pool '
                            'is not usable anymore')
            # Drain work queue and mark pending futures failed
            while True:
                try:
                    work_item = self._work_queue.get_nowait()
                except queue.Empty:
                    break
                if work_item is not None:
                    work_item.future.set_exception(BrokenThreadPool(self._broken))

    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True
            self._work_queue.put(NULL_ENTRY)
        
        if wait:
            for t in self._threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass
    shutdown.__doc__ = _base.Executor.shutdown.__doc__
