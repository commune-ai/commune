
import os
import sys
import time
import queue
import weakref
import itertools
import threading
import asyncio
from loguru import logger
from typing import Callable
from concurrent.futures._base import Future
import time
from tqdm import tqdm
from .task import Task, NULL_TASK
from .utils import new_event_loop, detailed_error, wait

class Executor:
    """Base threadpool executor with a priority queue"""

    # Used to assign unique thread names when thread_name_prefix is not supplied.
    _counter = itertools.count().__next__
    # submit.__doc__ = _base.Executor.submit.__doc__
    threads_queues = weakref.WeakKeyDictionary()

    def __init__(
        self,
        max_workers: int =None,
        maxsize : int = None ,
        thread_name_prefix : str ="",
        mode = 'thread',
    ):
        """Initializes a new Executor instance.
        Args:
            max_workers: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
        """
        self.start_time = time.time()
        max_workers = (os.cpu_count() or 1) * 5 if max_workers == None else max_workers
        maxsize = max_workers * 10 or None
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        self.mode = mode
        self.max_workers = max_workers
        self.task_queue = queue.PriorityQueue(maxsize=maxsize)
        self.idle_semaphore = threading.Semaphore(0)
        self.threads = []
        self.broken = False
        self.shutdown = False
        self.shutdown_lock = threading.Lock()
        self.thread_name_prefix = thread_name_prefix or ("Executor-%d" % self._counter() )

    @property
    def is_empty(self):
        return self.task_queue.empty()

    @property
    def is_full(self):
        return self.task_queue.full()

    def submit(self, 
               fn: Callable,
                params = None,
                args:dict=None, 
                kwargs:dict=None, 
                priority:int=1,
                timeout=200, 
                return_future:bool=True,
                wait = True, 
                path:str=None) -> Future:
        
        args = args or []
        kwargs = kwargs or {}
        if params != None:
            if isinstance(params, dict):
                kwargs = params
                if 'args' in kwargs and 'kwargs' in kwargs:
                    args = kwargs.pop('args')
                    kwargs = kwargs.pop('kwargs')
            elif isinstance(params, list):
                args = params
            else:
                raise ValueError(f"params must be a list or a dict {params, args, kwargs}")
        # check if the queue is full and if so, raise an exception
        if self.task_queue.full():
            if wait:
                while self.task_queue.full():
                    time.sleep(0.1)
            else:
                return {'success': False, 'msg':"cannot schedule new futures after maxsize exceeded"}
        with self.shutdown_lock:
            if self.broken:
                raise Exception("Executor is broken")
            if self.shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            task = Task(fn=fn, params=dict(args=args, kwargs=kwargs), timeout=timeout, path=path)
            # add the work item to the queue

            # if the function has a cost attribute, multiply the priority by the cost
            priority = kwargs.pop("priority", 1)
            if hasattr(fn, '__cost__'):
                priority = fn.__cost__ * priority

            self.task_queue.put((priority, task), block=False)
            # adjust the thread count to match the new task
            self.adjust_thread_count()
            
        # return the future (MAYBE WE CAN RETURN THE TASK ITSELF)
        if return_future:
            return task.future
        
        return task.future.result()

    def adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self.idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self.task_queue):
            q.put(NULL_TASK)

        num_threads = len(self.threads)
        if num_threads < self.max_workers:
            thread_name = "%s_%d" % (self.thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=self.worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self.task_queue,
                ),
            )
            t.daemon = True
            t.start()
            self.threads.append(t)
            self.threads_queues[t] = self.task_queue

    def shutdown(self, wait=True):
        with self.shutdown_lock:
            self.shutdown = True
            self.task_queue.put(NULL_TASK)
        if wait:
            for t in self.threads:
                try:
                    t.join(timeout=2)
                except Exception:
                    pass

    @classmethod
    def worker(cls, executor_reference, task_queue):
        new_event_loop()
        try:
            while True:
                work_item = task_queue.get(block=True)
                priority = work_item[0]

                if priority == sys.maxsize:
                    # Wake up queue management thread.
                    task_queue.put(NULL_TASK)
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
                    task_queue.put(NULL_TASK)
                    return
                del executor
        except Exception as e:
            e = detailed_error(e)

    @property
    def num_tasks(self):
        return self.task_queue.qsize()

    @property
    def is_empty(self):
        return self.task_queue.empty()
    
    def status(self):
        return dict(
            num_threads = len(self.threads),
            num_tasks = self.num_tasks,
            is_empty = self.is_empty,
            is_full = self.is_full
        )

    @classmethod
    def test(cls):
        def fn(x):
            result =  x*2
            print(result)
            return result
            
        self = cls()
        futures = []
        for i in range(10):
            futures += [self.submit(fn=fn, kwargs=dict(x=i))]
        for future in tqdm(futures):
            future.result()
        for i in range(10):
            futures += [self.submit(fn=fn, kwargs=dict(x=i))]

        results = wait(futures, timeout=10)
        
        while self.num_tasks > 0:
            print(self.num_tasks, 'tasks remaining')


        return {'success': True, 'msg': 'thread pool test passed'}
