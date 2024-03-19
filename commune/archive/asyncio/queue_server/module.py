import ray
import os,sys
from commune.block.ray.queue import Queue
from commune import Module
from commune.utils import dict_put,dict_get,dict_has,dict_delete
from copy import deepcopy
import asyncio


"""

Background Actor for Message Brokers Between Quees

"""

import threading
class AsyncQueueServer(Module):
    def __init__(self, loop=None, **kwargs):
        Module.__init__(self)
        # loop = asyncio.new_event_loop()
        # self.set_event_loop()
        # asyncio.set_event_loop(loop)
        # self.set_event_loop(loop=loop)
        nest_asyncio.apply()
        self.loop = asyncio.get_event_loop()
        self.queue = {}
    # def __del__(self):
    #     return self.loop.stop

    def create_queue(self, key:str, refresh=False, **kwargs):
        if self.queue_exists(key) and refresh:
            self.rm_queue(key)
        queue = asyncio.Queue(**kwargs)
        return self.add_queue(key=key, queue=queue)

    @staticmethod
    def new_event_loop(set_loop=False):
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop

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


    # async def async_put(self, key, value, *args, **kwargs):
    #     pass
        


    def put(self, key, value, sync=True, *args, **kwargs):

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


    def get(self, key, sync=True, *args, **kwargs):
        q = self.get_queue(key)
        job = q.get()
        if sync:
            return self.async_run(job)
        return job

    # async def get_batch(self, key, batch_size=10, sync=False, **kwargs):
    #     q = self.get_queue(key)
    #     batch_size = min(batch_size, q.qsize())
    #     jobs = [self.async_get(key, **kwargs) for i in range(batch_size)]
    #     return self.async_run(asyncio.gather(*job))





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

    async def bro(self):
        return 1

if __name__ == '__main__':
    import streamlit as st

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = AsyncQueueServer()
    st.write(server.put_batch('key', ['bro']*10, sync=True))
    st.write(server.put_batch('bro', ['bro']*10, sync=True))


    st.write(server.get_batch('key', batch_size=10, sync=True))
    # st.write(server.get('key', sync=True))

