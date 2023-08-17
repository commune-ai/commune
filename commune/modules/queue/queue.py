
import asyncio
import commune as c

class QueueServer(c.Module):
    def __init__(self, max_size=1000, mode = 'asyncio', **kwargs):
        self.queues = {}
        self.max_size = max_size
        self.mode = mode

    def queue_exists(self, key:str):
        return bool(key in self.queues)
        
    def add_queue(self, key:str, 
                  refresh:bool=False, 
                  max_size:int=None, 
                  mode:str=None, **kwargs):
        if mode == None:
            mode = self.mode
        if max_size == None:
            max_size = self.max_size
        
        max_size = self.max_size if max_size is None else max_size

        if mode == 'asyncio':
            self.queues[key] = asyncio.Queue(max_size,**kwargs)
        else:
            raise NotImplemented
            
        return self.queues[key]


    def get(self, key:str, sync=False, **kwargs):
        q = self.queues[key]
        job = q.get(**kwargs)
        return c.gather(job)
    
    def put(self, key, value, sync=False, **kwargs):
        q = self.queues[key]
        job = q.put(value, **kwargs)
        if future:
            return job
        return  c.gather(job)


    def get_batch(self, key, batch_size=10, sync=False, **kwargs):
        q = self.queues[key]
        batch_size = min(batch_size, q.qsize())
        jobs = [self.get(key, sync=False) for i in range(batch_size)]
        job = asyncio.gather(*jobs)
        if sync:
            return self.async_run(job)
        return job

    def size(self, key):
        # The size of the queue
        return self.queues[key].qsize()

    def empty(self, key):
        # Whether the queue is empty.
        return self.queues[key].empty()

    def full(self, key):
        # Whether the queue is full.
        return self.queues[key].full()

    def size_map(self):
        return {k: self.size(k) for k in self.queues}


    def test(self):
        for i in range(100):
            self.add_queue(i)
            for j in range(100):
                self.put(i, j)
                assert self.size(i) == j + 1, f'{self.size(i)} != {j + 1}'

            size = self.size(i)
            for j in range(100):
                self.get(i)
                assert self.size(i) == size - j -1, f'{self.size(i)} != {size - j -1}'

        return  {'success': True, 'message': 'QueueServer test passed'}
            