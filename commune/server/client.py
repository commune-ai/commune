

from concurrent.futures import ThreadPoolExecutor
import grpc
import json
import traceback
import threading
import uuid
import sys
import grpc
from types import SimpleNamespace
from typing import Tuple, List, Union
from grpc import _common
import sys
import os
import asyncio
from copy import deepcopy
import commune
from .serializer import Serializer



class VirtualModule:
    def __init__(self, module: str ='ReactAgentModule', include_hiddden: bool = False):
        '''
        VirtualModule is a wrapper around a Commune module.
        
        Args:
            module (str): Name of the module.
            include_hiddden (bool): If True, include hidden attributes.
        '''
        if isinstance(module, str):
            import commune
            self.module_client = commune.connect(module)
        else:
            self.module_client = module
        self.sync_module_attributes(include_hiddden=include_hiddden)
      
    def remote_call(self, fn: str, *args, **kwargs):
        
        
        return self.module_client(fn=fn, args=args, kwargs=kwargs)
            
    def sync_module_attributes(self, include_hiddden: bool = False):
        '''
        Syncs attributes of the module with the VirtualModule instance.
        
        Args:
            include_hiddden (bool): If True, include hidden attributes.
        '''
        from functools import partial
        
        for attr in self.module_client.whitelist_functions:
            # continue if attribute is private and we don't want to include hidden attributes
            if attr.startswith('_') and (not include_hiddden):
                continue
            
            # set attribute as the remote_call
            setattr(self, attr,  partial(self.remote_call, attr))
                
                
    # def __getattr__(self, key):
    #     return  self.module_client(fn='getattr', args=[key])


class Client( Serializer):
    """ Create and init the receptor object, which encapsulates a grpc connection to an axon endpoint
    """
    default_ip = '0.0.0.0'
    
    
    def __init__( 
            self,
            ip: str ='0.0.0.0',
            port: int = 80 ,
            max_processes: int = 1,
            timeout:int = 20,
            whitelist_functions: List[str] = ['functions', 'function_schema_map'],
            loop = None
        ):
        

        # Get endpoint string.
        self.ip = ip if ip else self.default_ip
        self.port = port
        try:
            self.loop = loop if loop else asyncio.get_event_loop()
        except RuntimeError as e:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        channel = grpc.aio.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000)])
        from .proto  import ServerStub
        stub = ServerStub( channel )

        self.channel = channel
        self.stub = stub
        self.client_uid = str(uuid.uuid1())
        self.semaphore = threading.Semaphore(max_processes)
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY
        self.sync_the_async()
        
        self.whitelist_functions = list(set(self(fn='functions', args=[False]) + whitelist_functions))


    @property
    def endpoint(self):
        return f"{self.ip}:{self.port}"

    def __call__(self, *args, **kwargs):
        try:
            return self.loop.run_until_complete(self.async_forward(*args, **kwargs))
        except TypeError:
            return self.loop.run_until_complete(self.async_forward(*args, **kwargs))
    def __str__ ( self ):
        return "Client({})".format(self.endpoint) 
    def __repr__ ( self ):
        return self.__str__()
    def __del__ ( self ):
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
                loop = asyncio.get_event_loop()
                loop.run_until_complete ( self.channel.close() )
        except:
            pass    
    def __exit__ ( self ):
        self.__del__()

    def nonce ( self ):
        import time as clock
        r"""creates a string representation of the time
        """
        return clock.monotonic_ns()
        
    def state ( self ):
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"

    def close ( self ):
        self.__exit__()

    def sign(self):
        return 'signature'

    async def async_forward(
        self, 
        data: object = None, 
        metadata: dict = None,
        timeout: int = 20,
        results_only = True,
        verbose=True,
        **kwargs
    ) :
        data = data if data else {}
        metadata = metadata if metadata else {}
        
        # the deepcopy is a hack to get around the fact that the data is being modified in place LOL
        kwargs, data, metadata = deepcopy(kwargs), deepcopy(data), deepcopy(metadata)
        
        data.update(kwargs)

        

        try:
            grpc_request = self.serialize(data=data, metadata=metadata)

            asyncio_future = self.stub.Forward(request = grpc_request, timeout = timeout)
            response = await asyncio_future
            response = self.deserialize(response)
            
            if results_only:
                try:
                    return response['data']['result']
                except Exception as e:
                    print(response) 
        except grpc.RpcError as rpc_error_call:
            response = str(rpc_error_call)
            commune.log(f"Timeout Error: {response}", verbose=verbose,color='red')

        # =======================
        # ==== Timeout Error ====
        # =======================
        except asyncio.TimeoutError:
            response = str(rpc_error_call)
            commune.log(f"Timeout Error: {response}", verbose=verbose,color='red')
    
        # ====================================
        # ==== Handle GRPC Unknown Errors ====
        # ====================================
        except Exception as e:
            response = str(e)
            commune.log(f"GRPC Unknown Error: {response}", color='red')
        return  response

    def sync_the_async(self):
        for f in dir(self):
            if 'async_' in f:
                setattr(self, f.replace('async_',  ''), self.sync_wrapper(getattr(self, f)))

    def sync_wrapper(self,fn:'asyncio.callable') -> 'callable':
        '''
        Convert Async funciton to Sync.

        Args:
            fn (callable): 
                An asyncio function.

        Returns: 
            wrapper_fn (callable):
                Synchronous version of asyncio function.
        '''
        def wrapper_fn(*args, **kwargs):
            return self.loop.run_until_complete(fn(*args, **kwargs))
        return  wrapper_fn

    def test_module(self):
        module = Client(ip='0.0.0.0', port=8091)
        import torch
        data = {
            'bro': torch.ones(10,10),
            'fam': torch.zeros(10,10)
        }

        st.write(module.forward(data=data))


    def virtual(self):
        return VirtualModule(module = self)        


if __name__ == "__main__":
    Client.test_module()

    # st.write(module)
