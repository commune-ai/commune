
import os
import sys
import threading
import asyncio
import nest_asyncio


class ThreadManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_threads=None):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: The maximum number of threads that can be used to
                execute the given calls.
            thread_name_prefix: An optional name prefix to give our threads.
            initializer: An callable used to initialize worker threads.
            initargs: A tuple of arguments to pass to the initializer.
        """
        self.max_threads = max_threads
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = []
        self._shutdown_lock = threading.Lock()
        self._shutdown = False

    def submit(self, fn, args=[],kwargs={}):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            
            thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            self._threads.append(thread)

        return thread


    @property
    def threads(self):
        return self._threads

    def __del__(self):
        self.shutdown()

    def shutdown(self, wait=True):
        if wait:
            for t in self._threads:
                try:
                    t.join()
                except Exception:
                    pass


if __name__ == '__main__':
    import streamlit as st


    async def bro(queue):
        while not queue.full():
            queue.put('bro')

    def fn(loop=None, queue=None):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bro(queue))


    manager = ThreadManager()

    loop = asyncio.new_event_loop()


    import queue
    queue = queue.Queue(maxsize=10)

    manager.submit(fn=fn, kwargs=dict(loop=loop, queue=queue))

    st.write(manager.threads[0])

    st.write(queue.__dict__)
    manager.shutdown()
    for i in range(10):
        st.write(queue.get())


    # task = asyncio.run_coroutine_threadsafe(bro(), loop)
    # st.write(task.result())
    loop.stop()


