

from typing import Tuple, List, Union
import sys
import os
import asyncio
import requests
from functools import partial
import commune as c
import aiohttp

class Client(c.Module):

    def __init__( 
            self,
            ip: str ='0.0.0.0',
            port: int = 50053 ,
            virtual: bool = True,
            **kwargs
        ):
        self.loop = c.get_event_loop()
        self.set_client(ip =ip,port = port)
        self.serialzer = c.module('serializer')()

    def set_client(self,
            ip: str =None,
            port: int = None ,
            ):
        self.ip = ip if ip else c.default_ip
        self.port = port if port else c.free_port() 

        c.print(f"Connecting to {self.ip}:{self.port}", color='green')
        self.address = f"{self.ip}:{self.port}"
       

    def resolve_client(self, ip: str = None, port: int = None) -> None:
        if ip != None or port != None:
            self.set_client(ip =ip,port = port)


    async def async_forward(self,
        fn,
        args = None,
        kwargs = None,
        ip: str = None,
        port : int= None,
        timeout: int = 4,
        return_error: bool = False,
         **extra_kwargs):

        self.resolve_client(ip=ip, port=port)
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        url = f"http://{self.address}/{fn}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, json= { "args": args,"kwargs": kwargs,}) as response:
                    response = await asyncio.wait_for(response.json(), timeout=timeout)
        except Exception as e:
            if return_error:
                response = {'error': str(e)}
            else: 
                raise e

        return response

    
    def forward(self,*args,return_future=False, **kwargs):
        forward_future =  self.async_forward(*args, **kwargs)
        if return_future:
            return forward_future
        else:
            # asyncio.wait_for(forward_future, timeout=timeout)

            return self.loop.run_until_complete(forward_future)
        
    __call__ = forward

    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()
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

    

    def sync_the_async(self, loop = None):
        for f in dir(self):
            if 'async_' in f:
                setattr(self, f.replace('async_',  ''), self.sync_wrapper(getattr(self, f), loop=loop))

    def sync_wrapper(self,fn:'asyncio.callable', loop = None) -> 'callable':
        '''
        Convert Async funciton to Sync.

        Args:
            fn (callable): 
                An asyncio function.

        Returns: 
            wrapper_fn (callable):
                Synchronous version of asyncio function.
        '''
        loop = loop if loop else self.loop
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

    def virtual(self):
        return c.virtual_client(module = self)


