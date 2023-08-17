

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
            network: bool = None,
            key = None,
            stream = False,
            **kwargs
        ):
        self.loop = c.get_event_loop()
        self.set_client(ip =ip,port = port)
        self.serializer = c.serializer()
        self.key = c.get_key(key)
        self.my_ip = c.ip()
        self.stream = stream
        self.network = c.resolve_network(network)
        self.start_timestamp = c.timestamp()

    
    def age(self):
        return  self.start_timestamp - c.timestamp()

    def set_client(self,
            ip: str =None,
            port: int = None ,
            verbose: bool = False
            ):
        self.ip = ip if ip else c.default_ip
        self.port = port if port else c.free_port() 
        if verbose:
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
        asyn: bool = True,
        full : bool = False,
        headers : dict ={'Content-Type': 'application/json'},
         **extra_kwargs):

        self.resolve_client(ip=ip, port=port)
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        url = f"http://{self.address}/{fn}/"
        request_data =  { 
            
                        "args": args,
                         "kwargs": kwargs,
                         "ip": self.my_ip,
                         "timestamp": c.timestamp(),
                         }

        
        request_data = self.serializer.serialize( request_data)
        request = c.copy(self.key.sign(request_data, return_json=True))

        assert self.key.verify(request), f"Request not signed with correct key"
        try:
            if asyn == True:
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(url, json=request, headers=headers) as response:
                            response = await asyncio.wait_for(response.json(), timeout=timeout)
                    except Exception as e:
                        async with session.post(url.replace(self.ip, '0.0.0.0'), json=request, headers=headers) as response:
                            response = await asyncio.wait_for(response.json(), timeout=timeout)
            else:
                response = requests.post(url,json=request, headers=headers)
                response = response.json()
      
            assert self.key.verify(response), f"Response not signed with correct key"
            response['data'] = self.serializer.deserialize(response['data'])
            response = response if full else response['data']
            if 'server_result' in response:
                response = response['server_result']
            
        except Exception as e:
            if return_error:
                response = c.detailed_error(e)
                return response
                
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


