""" Manages a pool of grpc connections as modules
"""

import math
from typing import Tuple, List, Union
from threading import Lock
import streamlit as st
import asyncio
from loguru import logger
import concurrent
import commune
from concurrent.futures import ThreadPoolExecutor
import commune as c
import asyncio

class ModulePool (c.Module):
    """ Manages a pool of grpc connections as modules
    """
    

    
    def __init__(
        self, 
        modules = None,
        max_modules:int = 20,
        stats = None,
        
    ):
        self.cull_mutex = Lock()
        self.modules = {}
        self.max_modules = max_modules
        self.add_modules(modules)
        self.set_stats(stats)
        self.total_requests = 0
        
        
    def set_stats(self, stats=None):
        if stats == None:
            stats = self.munch({})
        self.stats = stats
        return stats
    def add_module(self, *args, **kwargs)-> str:
        loop = self.get_event_loop()
        return loop.run_until_complete(self.async_add_module( *args, **kwargs))
        
    async def async_add_module(self, module:str = None, timeout=3)-> str:
        print(module)
        self.modules[module] = await c.async_connect(module, timeout=timeout,virtual=False)
        return self.modules[module]
    
    
    def add_modules(self, modules:list):
        if modules == None:
            modules = c.servers()
        if isinstance(modules, str):
            modules = c.servers(modules)
        loop = self.get_event_loop()
        return loop.run_until_complete(asyncio.gather(*[self.async_add_module(m) for m in modules]))
          
        
    
    def has_module(self, module:str)->bool:
        return bool(module in self.modules)
    
    def get_module(self, *args, **kwargs):
        loop = self.get_event_loop()
        return loop.run_until_complete(self.async_get_module(*args, **kwargs))




    async def async_get_module( self, 
                               module = None,
                               timeout=1, 
                               retrials=2) -> 'c.Client':
        
        if module == None:
             module = c.choice(list(self.modules.values()))
        elif isinstance(modules, str):
            if module in self.modules :
                module = self.modules[module]
            else:
                module =  await self.async_add_module(module)
        else:
            raise NotImplemented(module)
            
        return module
        
    

    def __str__(self):
        return "ModulePool({},{})".format(len(self.modules), self.max_modules)

    def __repr__(self):
        return self.__str__()
    
    # def __exit__(self):
    #     for module in self.modules:
    #         module.__del__()

    def forward (self, *args, **kwargs)  :

        loop = self.get_event_loop()
        return loop.run_until_complete (self.async_forward(*args, **kwargs) )


    async def async_forward (
            self, 
            fn:str,
            args:list = None,
            kwargs:dict = None, 
            module:list = None,
            timeout: int = 2,
        ) :
        # Init modules.
        
    
    
        module = await self.async_get_module( module )


        kwargs = {} if kwargs == None else kwargs
        args = [] if args == None else args
        
        result = await module.async_forward(fn=fn, args=args, kwargs=kwargs, timeout=timeout)
        
        

        return result


    async def async_forward_pool (
            self, 
            fn:str,
            args:list = None,
            kwargs:dict = None, 
            modules:list = None,
            timeout: int = 2,
            min_successes: int = 2,
        ) :
        # Init modules.
        
    
    
        module = await self.async_get_module( modules )


        kwargs = {} if kwargs == None else kwargs
        args = [] if args == None else args


        outputs = []
        
        while len(running_tasks) > 0:
            
            finished_tasks, running_tasks  = await asyncio.wait( running_tasks , return_when=asyncio.FIRST_COMPLETED)
            finished_tasks, running_tasks = list(finished_tasks), list(running_tasks)

            responses = await asyncio.gather(*finished_tasks)

            for response in responses:
                if  min_successes > 0:
                    if  response[1][0] == 1:
                        outputs.append( response )
                    if len(outputs) >= min_successes :
                        # cancel the rest of the tasks
                        [t.cancel() for t in running_tasks]
                        running_tasks = [t for t in running_tasks if t.cancelled()]
                        assert len(running_tasks) == 0, f'{len(running_tasks)}'
                        break
                else:
                    
                    outputs.append( response)

        return outputs

    @classmethod
    def test(cls, **kwargs):
        self = cls(modules='module')
        t = c.time()
        for i in range(10):
            print(i)
            output = self.forward('namespace')
            cls.print(output)
            
        cls.print('time', c.time() - t)
    
    
if __name__ == '__main__':
    ModulePool.run()