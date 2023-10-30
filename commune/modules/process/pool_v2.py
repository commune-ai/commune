
import os
import sys
import time
import queue
import random
import weakref
import itertools
import multiprocessing as mp
from loguru import logger
from typing import Callable
import concurrent
from concurrent.futures._base import Future
import commune as c


# Workers are created as daemon processs. This is done to allow the interpreter
# to exit when there are still idle processs in a ThreadPoolExecutor's process
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
# processs finish.

Task = c.module('executor.task')
NULL_ENTRY = (sys.maxsize, Task(None, (), {}))

class PoolTaskExecutor(c.Module):
    """Base processpool executor with a priority queue"""

    # Used to assign unique process names when process_name_prefix is not supplied.
    _counter = itertools.count().__next__
    # submit.__doc__ = _base.Executor.submit.__doc__
    process_queues = weakref.WeakKeyDictionary()

    def __init__(
        self,
        maxsize : int =-1,
        max_workers: int =None,
        process_name_prefix : str ="",
    ):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_workers: The maximum number of processs that can be used to
                execute the given calls.
            process_name_prefix: An optional name prefix to give our processs.
        """

        max_workers = (os.cpu_count()) if max_workers is None else max_workers
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
            
        self.max_workers = max_workers
        self.work_queue = mp.Queue(maxsize=maxsize)
        self.idle_semaphore = mp.Semaphore(0)
        self.processes = []
        self.broken = False
        self.shutdown = False
        self.shutdown_lock = mp.Lock()
        self.process_name_prefix = process_name_prefix or ("ProcessPoolExecutor-%d" % self._counter() )

    @property
    def is_empty(self):
        return self.work_queue.empty()

    
    def submit(self, fn: Callable, args=None, kwargs=None, timeout=200) -> Future:
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
            task = Task(fn=fn, args=args, kwargs=kwargs, timeout=timeout)
            # add the work item to the queue
            self.work_queue.put((priority, task), block=False)
            # adjust the process count to match the new task
            self.adjust_process_count()
            
            # return the future (MAYBE WE CAN RETURN THE TASK ITSELF)
            return task.future

    def adjust_process_count(self):
        if self.idle_semaphore.acquire(timeout=0):
            return


        num_processes = len(self.processes)
        if num_processes < self.max_workers:
            p = mp.Process(target=self.worker, args=(self.work_queue))
            p.daemon = True
            p.start()
            self.processes.append(p)
            self.process_queues[p] = self.work_queue

    def shutdown(self, wait=True):
        with self.shutdown_lock:
            for i in range(len(self.processes)):
                p = self.processes.pop()
                for _ in range(2):
                    self.work_queue.put(NULL_ENTRY)
                p.terminate()
                p.join()
            


    @staticmethod
    def worker(work_queue):
        
        try:
            while True:
                work_item = work_queue.get(block=True)
                priority = work_item[0]

                if priority == sys.maxsize:
                    # Wake up queue management process.
                    work_queue.put(NULL_ENTRY)
                    break

                item = work_item[1]
                item.run()
                del item

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


        return {'success': True, 'msg': 'process pool test passed'}
    
    def __del__(self):
        self.shutdown(wait=False)

        