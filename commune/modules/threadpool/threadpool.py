
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ThreadPoolExecutor."""

__author__ = 'Brian Quinlan (brian@sweetapp.com)'



import os
import sys
from concurrent.futures import _base
import itertools
import queue
import random
import threading
import weakref
import time
from loguru import logger
import commune as c


class WorkItem:
    def __init__(self, future, fn, start_time:int, args:list, kwargs:dict, timeout:int=10):
        self.future = future
        self.fn = fn
        self.start_time = start_time
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

    def run(self):
        """ Run the given work item
        """
        # Checks if future is canceled or if work item is stale
        if (not self.future.set_running_or_notify_cancel()) or (time.time()-self.start_time > self.timeout):
            return

        try:
            result = self.fn(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)



class PriorityThreadPoolExecutor(c.Module,_base.Executor):
    """ Base threadpool executor with a priority queue 
    """
    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__
    _threads_queues = weakref.WeakKeyDictionary()
    _shutdown = False
    NULL_ENTRY = (sys.maxsize, WorkItem(None, None, time.time(), (), {}))


    def __init__(self, maxsize = -1, max_workers=None, thread_name_prefix=''):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """

        self._max_workers = (os.cpu_count() or 1) * 5 if max_workers == None else max_workers
        assert self._max_workers > 0, "max_workers must be greater than 0"
        assert maxsize > 0 or maxsize == -1, "maxsize must be greater than 0 or -1"
        self._work_queue = queue.PriorityQueue(maxsize = maxsize)
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = set()
        self._broken = False
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread_name_prefix = (thread_name_prefix or
                                    ("ThreadPoolExecutor-%d" % self._counter()))
        self.timeout = 10

    def submit(self, fn, *args, timeout:str, **kwargs):
        with self._shutdown_lock:
            if self._broken:
                raise _base.BrokenExecutor(self._broken)

            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')

            priority = kwargs.get('priority', random.randint(0, 1000000))
            if priority == 0:
                priority = random.randint(1, 100)
            eplison = random.uniform(0,0.01) * priority
            start_time = time.time()
            if 'priority' in kwargs:
                del kwargs['priority']
            

            f = _base.Future()
            w = WorkItem(f, 
                        fn=fn, 
                        start_time=start_time,
                        args=args, kwargs=kwargs, 
                        timeout=self.timeout)
            self._work_queue.put((-float(priority + eplison), w), block=False)
            self._adjust_thread_count()
            return f

    @property
    def is_empty(self):
        return self._work_queue.empty()


    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(self.NULL_ENTRY)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=self._worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue))
            t.daemon = True
            t.start()
            self._threads.add(t)
            self._threads_queues[t] = self._work_queue


    def shutdown(self, wait=True):
        with self._shutdown_lock:
            self._shutdown = True
            self._work_queue.put(self.NULL_ENTRY)
        
        if wait:
            for t in self._threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass
    @classmethod
    def _worker(cls, executor_reference, work_queue):

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
                    work_queue.put(cls.NULL_ENTRY)
                    return
                del executor
        except BaseException:
            logger.error('work_item', work_item)
            _base.LOGGER.critical('Exception in worker', exc_info=True)


    def test(self, n=10, timeout=1):
        def test_fn():
            time.sleep(timeout+1)
            return 1

        futures = []
        for _ in range(n):
            f = self.submit(test_fn, timeout=timeout)
            futures.append(f)
        



        return f.result()
    @property
    def num_jobs(self):
        return self._work_queue.qsize()
    
    @staticmethod
    def gather(futures, timeout:int=None):
        """ Gather results from futures"""
        results = []
        if timeout != None:
            start_time = time.time()
        while len(futures) > 0:
            for f in futures:
                if f.done():
                    futures.remove(f)
                    results.append(f.result())
            if timeout != None:
                if time.time() - start_time > timeout:
                    break
        return results

    @staticmethod
    def get_done(futures):
        """ Gather results from futures"""
        done_futures = []
        while len(futures) > 0:
            for f in futures:
                if f.done():
                    done_futures.append(f)
        return done_futures

    @staticmethod
    def get_running(futures):
        """ Gather results from futures"""
        done_futures = []
        while len(futures) > 0:
            for f in futures:
                if not f.done():
                    done_futures.append(f)
        return done_futures



            
