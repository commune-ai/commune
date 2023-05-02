import queue
import commune
import threading
import multiprocessing

class Pool(commune.Module):
    def __init__(self, **kwargs):
        self.queue = self.munch({})
        self.init(**kwargs)
        
        
        
    def resolve_queue_name(self, name = None):
        if name is None:
            name = f'q{len(self.workers)}'
            
        assert isinstance(name, str)
        assert name not in self.queue
        
        return name
        
        
    def add_queue(self, name = None, mode='thread'):
        if mode == 'thread':
            self.queue[name] = queue.Queue()
        else:
            raise NotImplemented(mode)
        
    def add_queues(self, *names, mode='thread'):
        for name in names:
            self.add_queue(name, mode=mode)
        
    def resolve_worker_name(self, name = None):
        if name is None:
            name = f'w{len(self.workers)}'
            
        assert isinstance(name, str)
        assert name not in self.workers
        
        return name
      
    def add_workers(self, *names):
        for name in names:
            self.add_worker(name)
    def add_worker(self, name = None):
        name = self.resolve_worker_name(name)
        self.workers = self.config.get('workers', {})
        
        
        if not hasattr(self, 'queue'):
            queue = queue.Queue()
        if self.mode == 'thread': 
            if 'input' not in self.queue:
                self.queue['input'] = queue.Queue()
                
            t = threading.Thread(target=self.forward_requests)
            worker = t
            self.workers[name] = t
            t.start()
        elif self.mode == 'process':
            p = multiprocessing.Process(target=self.forward_requests)
            self.workers[name] = t
            p.start()
     
        else:
            raise ValueError("Invalid mode. Must be 'thread' or 'process'.")
    
    # write tests
    @classmethod
    def test(cls, **kwargs):
        self  = cls(**kwargs)
        self.add_queue('input')
        self.add_worker()
        self.add_request('input', 'test')
    
    def forward_requests(self):
        while self.run_loop:
            request = self.queue[''].get()
            if isinstance(request, str):
                if request == 'kill':
                    self.kill_loop()
                    break
                
            if isinstance(request, dict):
                fn = request['fn']
                kwargs = request['kwargs']
                args = request['args']
                print(request)
            # process request here
            print(f"Processing request: {request}")
            
    def kill_loop(self):
        self.run_loop = False

    def add_request(self, request):
        self.request_queue.put(request)
        
    def __del__(self):
        self.kill_loop()

if __name__ == "__main__":
    Pool.test()
    # Wait for all requests to be processed
    # queue.request_queue.join()

