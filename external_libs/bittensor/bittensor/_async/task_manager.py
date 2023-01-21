
import asyncio
from .base import AsyncBase

class AsyncTaskManager(AsyncBase):
    def __init__(loop=None):
        self.set_event_loop(loop)
        self.task_map = {}
    
    def submit(self, fn, args=[],  kwargs={}):
        self.loop.create_task(fn())