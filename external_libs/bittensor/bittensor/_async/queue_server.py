
import os,sys
from copy import deepcopy
import asyncio
from .base import AsyncBase
"""

Background Actor for Message Brokers Between Quees

"""
class AsyncQueueServer(AsyncBase) :

    def __init__(self, loop=None, **kwargs):
        self.set_event_loop(loop=loop)
        self.queue = {}

    def create_queue(self, key:str, refresh=False, **kwargs):
        if self.queue_exists(key) and refresh:
            self.rm_queue(key)
        queue = asyncio.Queue(**kwargs)
        return self.add_queue(key=key, queue=queue)

    def queue_exists(self, key:str):
        return bool(key in self.queue)

    def add_queue(self, key, queue: asyncio.Queue):
        assert isinstance(queue, asyncio.Queue)
        self.queue[key] = queue
        return self.queue[key]

    def rm_queue(self,key, *args, **kwargs):
        return self.queue.pop(key, None)
    
    def get_queue(self, key, *args, **kwargs):
        if not self.queue_exists(key):
            self.create_queue(key=key, *args, **kwargs)
        return self.queue[key]

    def list_queues(self, **kwargs):
        return list(self.queue.keys())
    
    ls = list_queues

    def get_batch(self, key, batch_size=10, sync=False, **kwargs):
        q = self.get_queue(key)
        batch_size = min(batch_size, q.qsize())
        jobs = [self.queue(key, sync=False) for i in range(batch_size)]
        job = asyncio.gather(*jobs)
        return jobs

    def put(self, key, value, sync=False, *args, **kwargs):
        q = self.get_queue(key,*args, **kwargs)
        job = q.put(value)
        if sync:
            return self.async_run(job)
        return job
    
    def put_batch(self, key:str, values: list, sync=True):
        assert isinstance(values, list)
        jobs = []
        for value in values:
            jobs += [self.put(key, value)]
        job = asyncio.gather(*jobs)  
        if sync :
            return self.async_run(job)
        return job

    def async_run(self, job):
        return self.loop.run_until_complete(job)

    def get(self, key, sync=False, **kwargs):
        q = self.get_queue(key)
        job = q.get()
        if sync:
            return self.async_run(job)
        return job

    def get_batch(self, key, batch_size=10, sync=False, **kwargs):
        q = self.get_queue(key)
        batch_size = min(batch_size, q.qsize())
        jobs = [self.get(key, sync=False) for i in range(batch_size)]
        job = asyncio.gather(*jobs)
        if sync:
            return self.async_run(job)
        return job

    def delete_all(self,  *args, **kwargs):
        for key in self.queue.keys():
            self.rm_queue(key, *args, **kwargs)

    rm_all = delete_all

    def size(self, key):
        # The size of the queue
        return self.queue[key].qsize()

    def empty(self, key):
        # Whether the queue is empty.
        return self.queue[key].empty()

    def full(self, key):
        # Whether the queue is full.
        return self.queue[key].full()

    def size_map(self):
        return {k: self.size(k) for k in self.queue}

