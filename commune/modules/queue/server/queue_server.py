
import asyncio
import commune as c

class QueueServer(c.Module):
    def __init__(self, max_size=1000, mode = 'asyncio', **kwargs):
        self.q = {}
        self.max_size = max_size
        self.mode = mode

    def queue_exists(self, key:str):
        return bool(key in self.q)
        
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
            self.q[key] = asyncio.Queue(max_size,**kwargs)
        else:
            raise NotImplemented
            
        return self.queue_map[key]
    
    def get(self, *args, **kwargs):
        return c.gather(self.async_get(*args, **kwargs))
    async def async_get(self, key:str, **kwargs):
        q = self.get_queue(key)
        return q.get(key, **kwargs)
    
    async def async_put(self, key, value, sync=True, *args, **kwargs):
        q = self.get_queue(key,*args, **kwargs)
        job = q.put(value)
        return job
    def put(self, *args, **kwargs):
        return c.gather(self.async_put(*args, **kwargs))

    def async_run(self, job):
        return self.loop.run_until_complete(job)



    def get_batch(self, key, batch_size=10, sync=False, **kwargs):
        q = self.get_queue(key)
        batch_size = min(batch_size, q.qsize())
        jobs = [self.get(key, sync=False) for i in range(batch_size)]
        job = asyncio.gather(*jobs)
        if sync:
            return self.async_run(job)
        return job

    def size(self, key):
        # The size of the queue
        return self.queue_map[key].qsize()

    def empty(self, key):
        # Whether the queue is empty.
        return self.queue_map[key].empty()

    def full(self, key):
        # Whether the queue is full.
        return self.queue_map[key].full()

    def size_map(self):
        return {k: self.size(k) for k in self.queue_map}