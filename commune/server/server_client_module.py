

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
sys.path.append(os.getenv('PWD'))
asyncio.set_event_loop(asyncio.new_event_loop())

import bittensor
import commune
from commune.proto import DataBlock
from commune.serializer import SerializerModule
from commune.server import ServerModule
import streamlit as st
import cortex

class ServerClientModule(nn.Module, SerializerModule):
    """ Create and init the receptor object, which encapsulates a grpc connection to an axon endpoint
    """
    
    def __init__( 
            self,
            ip: str ='localhost',
            port: int = 80 ,
            max_processes: 'int' = 1,
        ):

        super().__init__()
        # Get endpoint string.
        ip = ip if ip else 'localhost'
        self.endpoint = ip + ':' + str(port)


        channel = grpc.aio.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000)])
        stub = commune.grpc.CommuneStub( channel )

        self.loop = asyncio.get_event_loop()
        self.channel = channel
        self.stub = stub
        self.client_uid = str(uuid.uuid1())
        self.semaphore = threading.Semaphore(max_processes)
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY
        self.sync_the_async()

    def __str__ ( self ):
        return "ServerClient({})".format(self.endpoint) 
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
        data: object , 
        metadata: dict = {},
        timeout: int = 10
    ) :

        grpc_request = self.serialize(data=data, metadata=metadata)

        try:
            asyncio_future = self.stub.Forward(request = grpc_request, timeout = timeout)
            response = await asyncio_future
            response = self.deserialize(response)
            # asyncio_future.cancel()
        except grpc.RpcError as rpc_error_call:

            response = str(rpc_error_call)

        # =======================
        # ==== Timeout Error ====
        # =======================
        except asyncio.TimeoutError:
            response = str(rpc_error_call)
        # ====================================
        # ==== Handle GRPC Unknown Errors ====
        # ====================================
        except Exception as e:
            response = str(e)


        return  response

    @classmethod
    def sync_the_async(self):
        for f in dir(self):
            if 'async_' in f:
                setattr(self, f.replace('async_',  ''), self.sync_wrapper(getattr(self, f)))

    @staticmethod
    def sync_wrapper(fn:'asyncio.callable') -> 'callable':
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
            return asyncio.run(fn(*args, **kwargs))
        return  wrapper_fn

if __name__ == "__main__":
    module = ServerClientModule(ip='0.0.0.0', port=8091)


    data = {
        'bro': torch.ones(10,10),
        'fam': torch.zeros(10,10)
    }

    st.write(module.forward(data=data))
    # st.write(module)
