""" Manages a pool of grpc connections as clients
"""

import math
from typing import Tuple, List, Union
from threading import Lock
import streamlit as st
import torch
import asyncio
from loguru import logger
import concurrent
import commune
from concurrent.futures import ThreadPoolExecutor
import commune
import asyncio
class ModulePool (commune.Module):
    """ Manages a pool of grpc connections as clients
    """
    

    
    def __init__(
        self, 
        modules,
        max_active_clients = 20,
        
    ):
        self.pool = {}
        self.add_modules(modules)
        self.max_active_clients = self.max_active_clients
        self.client_stats = {}
        self.cull_mutex = Lock()
        self.total_requests = 0
        
        
    def add_module(self, *args, **kwargs)-> str:
        loop = self.get_event_loop()
        return loop.run_until_complete(self.async_add_module( *args, **kwargs))
        
    async def async_add_module(self, module:str = None, timeout=3)-> str:
        if not hasattr(self, 'modules'):
            self.modules = {}
        self.modules[module] = await commune.async_connect(module, timeout=timeout)
        return module
    
    
    def add_modules(self, modules:list):
        
        return asyncio.gather(*[self.async_add_module(m) for m in self.modules])
          
        
    
    def has_module(self, module:str)->bool:
        return bool(module in self.modules)
    
    def get_module(self, module:str):
        if not self.has_module(module):
            self.add_module(module)
        return self.modules[module]

    async def async_get_module( self, 
                               module = None,
                               timeout=1, 
                               retrials=2) -> 'commune.Client':
        
        if module not in self.moddules :
            self.async_add_module(module)
            
        return self.module[ module ]
        
    

    def __str__(self):
        return "ModulePool({},{})".format(len(self.clients), self.max_active_clients)

    def __repr__(self):
        return self.__str__()
    
    def __exit__(self):
        for client in self.clients:
            client.__del__()

    def forward (
            self, 
            args = None,
            kwargs = None, 
            modules: List [str ] = None,
            min_successes: int = None,
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:

        loop = self.get_event_loop()
        return loop.run_until_complete (self.async_forward(kwargs=kwargs) )



    async def async_forward (
            self, 
            fn: None,
            module = None,
            args = None,
            kwargs = None,
            timeout: int = 2,
            min_successes: int = 2,
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        # Init clients.
        
    
    
        client = await self.async_get_module( module )


        kwargs = {} if kwargs == None else kwargs
        args = [] if args == None else args

        # Make calls.
        running_tasks = []
        for index, (client) in enumerate(clients.items()):
            args, kwargs = self.copy(args), self.copy(kwargs)
            task = asyncio.create_task(
                client.async_forward(*args, **kwargs)
            )
            running_tasks.append(task)


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

    def check_clients( self ):
        r""" Destroys clients based on QPS until there are no more than max_active_clients.
        """
        with self.cull_mutex:
            # ---- Finally: Kill clients over max allowed ----
            if len(self.clients) > self.max_active_clients:
                c = list(self.clients.keys())[0]
                self.clients.pop(c, None)   

    @classmethod
    def test(cls, **kwargs):
        return cls(modules='module')