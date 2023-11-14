
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

Task = c.module('executor.task')

NULL_ENTRY = (sys.maxsize, Task(None, (), {}))

class ThreadPoolExecutor(c.Module):
    """Base threadpool executor with a priority queue"""

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__
    # submit.__doc__ = _base.Executor.submit.__doc__
    threads_queues = weakref.WeakKeyDictionary()

    def __init__(
        self,
        max_workers: int =None,
        maxsize : int =-1,
        thread_name_prefix : str ="",
    ):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """

        max_workers = (os.cpu_count() or 1) * 5 if max_workers == None else max_workers
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
            
        self.max_workers = max_workers
        self.work_queue = queue.PriorityQueue(maxsize=maxsize)
        self.idle_semaphore = threading.Semaphore(0)
        self.threads = []
        self.broken = False
        self.shutdown = False
        self.shutdown_lock = threading.Lock()
        self.thread_name_prefix = thread_name_prefix or ("ThreadPoolExecutor-%d" % self._counter() )

    @property
    def is_empty(self):
        return self.work_queue.empty()

    
    def submit(self, fn: Callable, args:dict=None, kwargs:dict=None, timeout=200, return_future:bool=True, path:str=None) -> Future:
        args = args or ()
        kwargs = kwargs or {}
        with self.shutdown_lock:
            if self.broken:
                raise Exception("ThreadPoolExecutor is broken")

            if self.shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            priority = kwargs.get("priority", 1)
            if "priority" in kwargs:
                del kwargs["priority"]
            task = Task(fn=fn, args=args, kwargs=kwargs, timeout=timeout, path=path)
            # add the work item to the queue
            self.work_queue.put((priority, task), block=False)
            # adjust the thread count to match the new task
            self.adjust_thread_count()
            
        # return the future (MAYBE WE CAN RETURN THE TASK ITSELF)
        if return_future:
            return task.future
        else: 
            return task.future.result()


    def adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self.idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self.work_queue):
            q.put(NULL_ENTRY)

        num_threads = len(self.threads)
        if num_threads < self.max_workers:
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
            self.threads.append(t)
            self.threads_queues[t] = self.work_queue

    def shutdown(self, wait=True):
        with self.shutdown_lock:
            self.shutdown = True
            self.work_queue.put(NULL_ENTRY)
        if wait:
            for t in self.threads:
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
                if executor is None or executor.shutdown:
                    # Flag the executor as shutting down as early as possible if it
                    # is not gc-ed yet.
                    if executor is not None:
                        executor.shutdown = True
                    # Notice other workers
                    work_queue.put(NULL_ENTRY)
                    return
                del executor
        except Exception as e:
            c.print(e, color='red')
            c.print("work_item", work_item, color='red')

            e = c.detailed_error(e)
            c.print("Exception in worker", e, color='red')

    @property
    def num_tasks(self):
        return self.work_queue.qsize()

    @classmethod
    def as_completed(futures: list):
        assert isinstance(futures, list), "futures must be a list"
        return [f for f in futures if not f.done()]

    @staticmethod
    def wait(futures:list) -> list:
        futures = [futures] if not isinstance(futures, list) else futures
        results = []
        for future in c.as_completed(futures):
            results += [future.result()]
        return results

    
    @classmethod
    def test(cls):
        def fn(x):
            result =  x*2
            return result
            
        self = cls()
        futures = []
        for i in range(100):
            futures += [self.submit(fn=fn, kwargs=dict(x=i))]
        for future in c.tqdm(futures):
            future.result()
        for i in range(100):
            futures += [self.submit(fn=fn, kwargs=dict(x=i))]

        results = c.wait(futures)
        
        while self.num_tasks > 0:
            c.print(self.num_tasks, 'tasks remaining', color='red')


        return {'success': True, 'msg': 'thread pool test passed'}

        