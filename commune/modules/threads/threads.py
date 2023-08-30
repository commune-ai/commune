import threading
import time
import queue
from loguru import logger
import commune as c

class Threads(c.Module):
    r""" Manages the queue the producer thread that monitor and fills the queue.
    """
    max_threads = 10
    threads = {}



    @classmethod
    def add_thread(cls, fn:'callable', args = None, kwargs:dict = None, tag=None):
        args = args or []
        kwargs = kwargs or {}
        name = fn.__name__
        if tag != None:
            name = name + '::' + tag

        if len(cls.threads) > cls.max_threads:
            oldest = cls.oldest_thread()
            cls.rm_thread(oldest)
            c.print(f'removed oldest thread {oldest}')
        stop_event = threading.Event()
        thread = threading.Thread(target=cls.run, kwargs={'fn': fn, 'kwargs': kwargs, 'stop_event': stop_event})
        thread.daemon = True
        thread.start()
        c.print('thread started')
        cls.threads[name] = {'thread': thread, 'stop_event': stop_event, 'fn': fn, 'args': args, 'kwargs': kwargs, 'time': c.time()}


    @classmethod
    def oldest_thread(self):
        oldest = None
        for name in self.threads:
            if oldest == None:
                oldest = name
            elif self.threads[name]['time'] < self.threads[oldest]['time']:
                oldest = name
        return oldest

    # Create an event to signal thread termination

    @classmethod
    def rm_thread(cls, name):
        if name in cls.threads:
            stop_event = cls.threads[name]['stop_event']
            stop_event.set()
            del cls.threads[name]

            
    @classmethod
    def test(cls):
        # Example of how to use the Threads class
        def worker_function(arg1=1, tag=None):
            while True:
                print(f"Worker thread is running...{tag}")
                time.sleep(1)
                
        n = 100
        for tag in range(n):
            tag = str(tag)
            Threads.add_thread(worker_function, kwargs={'tag': tag}, tag=tag)

        # Let some time pass...

        # Removing a thread
        for tag in range(n):
            tag = str(tag)
            c.sleep(10)
            Threads.rm_thread(f'worker_function::{tag}')



        print("Main program finished.")


    def __del__(self):
        for name in self.threads:
            self.rm_thread(name)
