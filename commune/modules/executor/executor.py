
import os
import sys
import time
import queue
import random
import weakref
import argparse
import bittensor
import itertools
import threading

from loguru import logger
from typing import Callable
from concurrent.futures import _base
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


class WorkItem(object):
    def __init__(self, fn, args, kwargs, timeout:int=10):
        self.future = _base.Future()
        self.fn = fn
        self.start_time = time.time()
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

    def info(self):
        return {
            'fn': self.fn.__name__,
            'kwargs': self.kwargs,
            'args': self.args,
            'timeout': self.timeout
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
    


NULL_ENTRY = (sys.maxsize, WorkItem(None, (), {}))

class PriorityThreadPoolExecutor(c.Module):
    """Base threadpool executor with a priority queue"""

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__
    # submit.__doc__ = _base.Executor.submit.__doc__
    threads_queues = weakref.WeakKeyDictionary()

    def __init__(
        self,
        maxsize=-1,
        max_workers=None,
        thread_name_prefix="",
        initargs=(),
    ):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """

        max_workers = (os.cpu_count() or 1) * 5 if max_workers is None else max_workers
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
            
        self._max_workers = max_workers
        self.work_queue = queue.PriorityQueue(maxsize=maxsize)
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = set()
        self._broken = False
        self.shutdown = False
        self.shutdown_lock = threading.Lock()
        self.thread_name_prefix = thread_name_prefix or ("ThreadPoolExecutor-%d" % self._counter() )

    @property
    def is_empty(self):
        return self.work_queue.empty()

    def submit(self, fn: Callable, args=None, kwargs=None) -> _base.Future:
        args = args or ()
        kwargs = kwargs or {}
        with self.shutdown_lock:
            if self._broken:
                raise Exception("ThreadPoolExecutor is broken")

            if self.shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            priority = kwargs.get("priority", 1)
            start_time = time.time()
            if "priority" in kwargs:
                del kwargs["priority"]
            w = WorkItem(fn=fn, args=args, kwargs=kwargs)
            self.work_queue.put((priority, w), block=False)
            self.adjust_thread_count()
            return w.future


    def adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self.work_queue):
            q.put(NULL_ENTRY)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self.thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=self.worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self.work_queue,
                ),
            )
            t.daemon = True
            t.start()
            self._threads.add(t)
            self.threads_queues[t] = self.work_queue

    def shutdown(self, wait=True):
        with self.shutdown_lock:
            self.shutdown = True
            self.work_queue.put(NULL_ENTRY)

        if wait:
            for t in self._threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass

    @staticmethod
    def worker(executor_reference, work_queue):
        
        try:
            while True:
                work_item = work_queue.get(block=True)
                priority = work_item[0]

                if priority == sys.maxsize:
                    # Wake up queue management thread.
                    work_queue.put(NULL_ENTRY)
                    break

                item = work_item[1]

                if item is not None:
                    item.run()
                    # Delete references to object. See issue16284
                    del item
                    continue

                executor = executor_reference()
                # Exit if:
                #   - The interpreter is shutting down OR
                #   - The executor that owns the worker has been collected OR
                #   - The executor that owns the worker has been shutdown.
                if shutdown or executor is None or executor.shutdown:
                    # Flag the executor as shutting down as early as possible if it
                    # is not gc-ed yet.
                    if executor is not None:
                        executor.shutdown = True
                    # Notice other workers
                    work_queue.put(NULL_ENTRY)
                    return
                del executor
        except Exception as e:
            c.print("work_item", work_item, color='red')
            c.print("Exception in worker", e, color='red')

    @property
    def num_tasks(self):
        return self.work_queue.qsize()


    
    @classmethod
    def test(cls):
        def fn(x):
            result =  x*2
            return result
            
        self = cls()
        futures = []
        for i in range(100):
            futures += [self.submit(fn=fn, kwargs=dict(x=i))]

        for future in futures:
            c.print(future.result())
        while self.num_tasks > 0:
            c.print(self.num_tasks)


        return {'success': True, 'msg': 'thread pool test passed'}

        