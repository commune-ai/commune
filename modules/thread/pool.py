
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


class Task:
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
            data = self.fn(*self.args, **self.kwargs)
            self.future.set_result(data)
            self.status = 'done'
        except Exception as e:
            # what does this do? A: it sets the exception of the future, and sets the status to failed
            self.future.set_exception(e)
            self.status = 'failed'
            data = c.detailed_error(e)

   
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
        **kwargs
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

    
    def submit(self, fn: Callable, args=None, kwargs=None, timeout=200, return_future:bool=True) -> Future:
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

        