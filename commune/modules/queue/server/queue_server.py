
import asyncio
import commune as c

class QueueServer(c.Module):
    def __init__(self, max_size=1000, default_key = 'default', **kwargs):
        self.loop = c.get_event_loop()
        self.max_size = max_size
        self.queue_map = {}
        self.add_queue(default_key)


    def queue_exists(self, key:str):
        return bool(key in self.queue_map)
        
    def add_queue(self, key:str, refresh=False, max_size=None, **kwargs):
        max_size = self.max_size if max_size is None else max_size
        self.queue_map[key] = asyncio.Queue(max_size,**kwargs)
        return self.queue_map[key]

    def get_queue(self, key, max_queue_size=1000, *args, **kwargs):
        if key not in self.queue_map:
            self.add_queue(key, max_size=max_queue_size, *args, **kwargs)
        return self.queue_map[key]


    async def async_get(self, key, *args, **kwargs):
        q = self.get_queue(key)
        return q.get(*args, **kwargs)
    def get(self, *args, **kwargs):
        return self.loop.run_until_complete(self.async_get(*args, **kwargs))

    async def async_put(self, key, value, sync=True, *args, **kwargs):
        q = self.get_queue(key,*args, **kwargs)
        job = q.put(value)
        return job
    def put(self, *args, **kwargs):
        return self.loop.run_until_complete(self.async_put(*args, **kwargs))

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