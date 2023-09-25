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

class ClientPool (commune.Module):
    """ Manages a pool of grpc connections as clients
    """
    def __init__(
        self, 
        modules,
        max_active_clients = 20,
        
    ):
        
        self.add_modules(modules)
        self.max_active_clients = self.max_active_clients
        
        self.client_stats = {}
        if modules == None:
            modules = self.modules()
        self.cull_mutex = Lock()
        self.total_requests = 0


    def __str__(self):
        return "ClientPool({},{})".format(len(self.clients), self.max_active_clients)

    def __repr__(self):
        return self.__str__()
    
    def __exit__(self):
        for client in self.clients:
            client.__del__()

    def forward (
            self, 
            modules: List [str ] = None,
            args = None,
            kwargs = None, 
            timeout: int,
            min_successes: int = None,
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from x are sent forward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    TODO(const): Allow multiple tensors.
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                timeout (int):
                    Request timeout.

            Returns:
                forward_outputs (:obj:`List[ List[ torch.FloatTensor ]]` of shape :obj:`(num_endpoints * (num_synapses * (shape)))`, `required`):
                    Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

                forward_codes (:obj:`List[ List[bittensor.proto.ReturnCodes] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call return ops.

                forward_times (:obj:`List[ List [float] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call times
        """

        loop = self.get_event_loop()
        return loop.run_until_complete ( 
            self.async_forward(kwargs=kwargs) 
        )


    def add_module(self, module):
        if module in self.pool:
            return module
        
        self.pool[module] = commune.connect(module)
         
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
        
    
    
        client = await self.async_get_clients( module )


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
                    


    async def async_get_client( self, 
                               module = None,
                               timeout=1 ) -> 'commune.Client':
        r""" Finds or creates a client TCP connection associated with the passed Neuron Endpoint
            Returns
                client: (`commune.Client`):
                    client with tcp connection endpoint at endpoint.ip:endpoint.port
        """
        # ---- Find the active client for this endpoint ----
        
        modules = self.modules(module)
        
        
        if module == None:
            client = self.choice(self.clients.values())
        if module in self.clients :
            client = self.clients[module]
        else:
            client = await self.async_connect(module, timeout=timeout)
            self.clients[ client.endpoint.hotkey ] = client
            
        return client
    
