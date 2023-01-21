

from concurrent.futures import ThreadPoolExecutor
import grpc
import json
import traceback
import torch
import threading
import uuid
import sys
import torch.nn as nn
import grpc
import time as clock
from types import SimpleNamespace
from typing import Tuple, List, Union
from loguru import logger
from grpc import _common
import sys
import os
import asyncio
import bittensor
import tuwang
from .proto  import DataBlock, ServerStub
from .serializer import Serializer
from .server import Server
from copy import deepcopy
if os.getenv('USE_STREAMLIT'):
    import streamlit as st

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
            loop = None
        ):

        # Get endpoint string.
        self.ip = ip if ip else self.default_ip
        self.port = port
        self.loop = loop if loop else asyncio.get_event_loop()
        
        
        channel = grpc.aio.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000)])
        stub = ServerStub( channel )

        self.channel = channel
        self.stub = stub
        self.client_uid = str(uuid.uuid1())
        self.semaphore = threading.Semaphore(max_processes)
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY
        self.sync_the_async()

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
            raise(response)
        # =======================
        # ==== Timeout Error ====
        # =======================
        except asyncio.TimeoutError:
            response = str(rpc_error_call)
            raise(response)
    
        # ====================================
        # ==== Handle GRPC Unknown Errors ====
        # ====================================
        except Exception as e:
            response = str(e)
            raise e
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

        data = {
            'bro': torch.ones(10,10),
            'fam': torch.zeros(10,10)
        }

        st.write(module.forward(data=data))


if __name__ == "__main__":
    Client.test_module()

    # st.write(module)
