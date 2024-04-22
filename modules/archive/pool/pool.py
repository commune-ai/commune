import queue
import commune as c
import threading
import multiprocessing
import asyncio


class Pool(c.Module):
    def __init__(self, 
                 modules = None
                 **kwargs):

    
        self.run_loop = True
        self.init(**kwargs)
        self.set_fn(fn)
        self.add_workers()
        
        
    def set_fn(self, fn):
        if fn is None:
            fn = self.default_fn
        self.fn = fn
        assert callable(self.fn)
        return fn
        
    def resolve_queue_name(self, name = None):
        if name is None:
            name = f'Q::{len(self.workers)}'
            
        assert isinstance(name, str)
        
        return name
        
        
    def add_queue(self, name = None, mode='thread', update:bool=False):
        if not hasattr(self, 'queue'):
            self.queue = self.munch({})
        
        name = self.resolve_queue_name(name)
        if name in self.queue and not update:
            return self.queue[name]
        
        
        if mode == 'thread':
            self.queue[name] = queue.Queue()
        else:
            raise NotImplemented(mode)
        
    def add_queues(self, *names,  **kwargs):
        for name in names:
            self.add_queue(name, **kwargs)
        
    def resolve_worker_name(self, name = None):
        if name is None:
            name = f'W::{len(self.workers)}'
        else:
            name = str(name)
        assert isinstance(name, str)
        assert name not in self.workers
        return name
      
    def add_workers(self, *names, **kwargs):
        if len(names) == 0:
            names = [self.resolve_worker_name(i) for i in range(self.config.num_workers)]
        for name in names:
            self.add_worker(name, **kwargs)
        
        
        
    @property
    def workers(self):
        return self.config.get('workers', {})
    @workers.setter
    def workers(self, value):
        self.config['workers'] = value
        return value
    
    @property
    def mode(self):
        return self.config.get('mode', {})
    @workers.setter
    def mode(self, value):
        self.config['mode'] = value
        return value
        
        
    def resolve_mode(self, mode = None):
        if mode is None:
            mode = self.config.mode
        return mode
        
    def add_worker(self, name = None, 
                   update:bool= False,
                   fn = None,
                   mode:str=None,
                   in_queue:str=None,
                   out_queue:str=None,
                   verbose: bool = True):
        name = self.resolve_worker_name(name)
        mode = self.resolve_mode(mode)
        
        
        if name in self.workers and not update:
            return self.workers[name]
        
        queue_prefix = '::'.join(name.split('::')[:-1])
        kwargs = dict(
            in_queue = 'input' if in_queue is None else in_queue,
            out_queue = 'output' if out_queue is None else out_queue,   
            fn = self.fn if fn is None else fn,
        )

        self.add_queue(kwargs['in_queue'], mode=mode)
        self.add_queue(kwargs['out_queue'], mode=mode)
        
        if verbose:
            self.print(f"Adding worker: {name}, mode: {mode}, kwargs: {kwargs}")

        self.lock = threading.Lock()
        if self.config.mode == 'thread': 
        
            t = threading.Thread(target=self.forward_requests, kwargs=kwargs)
            worker = t
            self.workers[name] = t
            t.daemon = self.config.daemon
            t.start()
        elif self.config.mode == 'process':
            p = multiprocessing.Process(target=self.forward_requests, kwargs=kwargs)
            self.workers[name] = t
            p.start()
     
        else:
            raise ValueError("Invalid mode. Must be 'thread' or 'process'.")
    
    # write tests
    @classmethod
    def test(cls, **kwargs):
        self  = cls(**kwargs)
        for i in range(10):
            self.put(dict(module='dataset.bittensor', fn='sample'))  
        cls.print('Done')
        for  i in range(10):
            self.get()
        # self.kill()

    
    def default_fn(self,request, **kwargs):
        
        if isinstance(request, dict):
            # get request from queue
            # get function and arguments
            module = request.get('module') # identity function
            fn = request.get('fn', 'forward') # identity function
            kwargs = request.get('kwargs', {})
            args = request.get('args', [])
            assert callable(fn), f"Invalid function: {fn}"
            output = fn(*args, **kwargs)
            
            return output
        else:
            return request
            
    def forward_requests(self,  **kwargs):
        
        with self.lock:
            verbose = kwargs.get('verbose', True)
            in_queue = kwargs.get('in_queue', 'input')
            out_queue = kwargs.get('out_queue', 'output')
            fn = kwargs.get('fn', self.fn)
            name = kwargs.get('name', 'worker')
            worker_prefix = f"Worker::{name}"
        
            asyncio.set_event_loop(asyncio.new_event_loop())
            color= 'yellow'
            while True:
                request = self.queue[in_queue].get()
                if verbose:
                    self.print(f"{worker_prefix} <-- request: {request}", color='yellow')
                    
                if request == 'kill':
                    if verbose: 
                        self.print(f"Killing worker: {name}", color='red')
                    break
                
                try:
                    output = fn(request, **kwargs)
                except Exception as e:
                    output = str(e)
                    self.print(f"Error: {e}", color='red')
                    continue
                self.queue[out_queue].put(output)
                
                if verbose:
                    self.print(f"{worker_prefix} --> result: {output}", color='green')
    def kill_loop(self):
        self.run_loop = False

    def kill_workers(self):
        # kill all workers
        for name, worker in self.workers.items():
            self.add_request('kill')
        
    def __del__(self):
        self.kill_workers()
    def put(self, request):
        
        self.queue.input.put(request)

    def get(self):
        return getattr(self.queue, q).get()
        

if __name__ == "__main__":
    Pool.test()
    # Wait for all requests to be processed
    # queue.request_queue.join()

