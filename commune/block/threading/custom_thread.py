import threading

class CustomThread(threading.Thread):
    '''
    Custom thread for Running forever
    '''
    def __init__(self, fn, args=[], kwargs={}, forever:bool=False,  ):
        threading.Thread.__init__(self)
        self.fn = fn
        self.forever = forever
        self.kwargs = kwargs
        self.args = args
        self._stop_event = threading.Event()

    def run(self):
        # target function of the thread class

        while not self.stopped:
            self.fn(*self.args, **self.kwargs)
            if self.forever:
                continue
            else:
                self.stop()

    def stop(self):
        self._stop_event.set()
        assert self.stopped

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
  


class ThreadManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_threads:int=None):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: 
                The maximum number of threads that can be used to
                execute the given calls.
        """
        self.max_threads = max_threads
        self._idle_semaphore = threading.Semaphore(0)
        self._threads = []
        self._shutdown_lock = threading.Lock()
        self._shutdown = False


    def submit(self, fn, args:Optional[list]=[],kwargs:Optional[dict]={}) -> Any:
        '''
        Submit a function with args and kwargs on a seperate thread.

        Args
            fn (Callable):
                Function to place on the thread.
            args (list):
                Arguments to a function.
            kwargs (dict):
                Key word arguments to a function.
        '''
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            
            thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            self._threads.append(thread)

        return thread


    @property
    def threads(self):
        '''List threads.'''
        return self._threads

    def __del__(self):
        self.shutdown()

    def shutdown(self, wait=True):
        '''Shutdown threads'''
        if wait:
            for t in self._threads:
                # forces thread to stop
                t.raise_exception()
                t.join()
