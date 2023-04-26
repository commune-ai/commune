import queue
import threading
import multiprocessing

class RequestQueue:
    def __init__(self, 
                 mode='thread', 
                 num_workers=5,
                 run_loop=True,
                 num_workers=5
                 ):
        self.merge_kwargs(kwargs=locals())
        self.request_queue = queue.Queue()
        
        self.create_workers()

    def merge_kwargs(self, kwargs):
        kwargs.pop('self')
        self.__dict__.update(kwargs)
        
    def create_workers(self):
        self.workers = []
        if self.mode == 'thread':
            for i in range(5):
                t = threading.Thread(target=self._process_requests)
                self.workers.append(t)
                t.start()
        elif self.mode == 'process':
            for i in range(5):
                p = multiprocessing.Process(target=self._process_requests)
                self.workers.append(p)
                p.start()
        else:
            raise ValueError("Invalid mode. Must be 'thread' or 'process'.")
    
    def _process_requests(self):
        while self.run_loop:
            request = self.request_queue.get()
            if isinstance(request, dict):
                fn = request['fn']
                kwargs = request['kwargs']
                args = request['args']
            # process request here
            print(f"Processing request: {request}")
            
    def kill_loop(self):
        self.run_loop = False

    def add_request(self, request):
        self.request_queue.put(request)

queue = RequestQueue(mode='thread')

# Add some requests to the queue
for i in range(10):
    queue.add_request(f"Request {i+1}")

# Wait for all requests to be processed
queue.request_queue.join()

