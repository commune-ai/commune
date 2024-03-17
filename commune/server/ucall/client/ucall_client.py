

from typing import Tuple, List, Union
import sys
import os
import asyncio
import requests
from functools import partial
import commune as c
import aiohttp
import json
from ucall.client import Client
from aiohttp.streams import StreamReader


class ClientUcall(c.Module):

    def __init__( 
            self,
            ip: str ='0.0.0.0',
            port: int = 50053 ,
            network: bool = None,
            key : str = None,
            loop: 'asyncio.EventLoop' = None,
            use_http: bool = True
        ):
        self.loop = c.get_event_loop() if loop == None else loop
        
        
        
        self.ip = ip = ip if ip else c.default_ip
        self.port = port = port if port else c.free_port() 
        self.address = f"{self.ip}:{self.port}"
        self.use_http = use_http
        
        self.client = Client(uri = self.ip, port = self.port, use_http = use_http)

        self.set_client(ip =ip, port = port, use_http=use_http)
        self.serializer = c.serializer()
        self.key = c.get_key(key)
        self.my_ip = c.ip()
        self.network = c.resolve_network(network)
        self.start_timestamp = c.timestamp()
    
    def age(self):
        return  self.start_timestamp - c.timestamp()

       
    async def async_forward(self,
        fn: str,
        args: list = None,
        kwargs: dict = None,
        timeout: int = 10,
        headers : dict ={'Content-Type': 'application/json'}):

        args = args if args else []
        kwargs = kwargs if kwargs else {}

        request_data =  { 
                        "args": args,
                        "kwargs": kwargs,
                        "ip": self.my_ip,
                        "timestamp": c.timestamp(),
                        }

        # serialize this into a json string
        request_data = self.serializer.serialize( request_data)

        # sign the request
        params = self.key.sign(request_data, return_json=True)

        client = self.client.remote_call(fn=fn, input=params)


        return response


    def process_output(self, result):
        ## handles 
        if isinstance(result, str):
            result = json.loads(result)
        if 'data' in result:
            result = self.serializer.deserialize(result['data'])
            return result['data']
        else:
            return result
        
    def forward(self,*args,return_future:bool=False, timeout:str=4, **kwargs):
        forward_future = asyncio.wait_for(self.async_forward(*args, **kwargs), timeout=timeout)
        if return_future:
            return forward_future
        else:
            return self.loop.run_until_complete(forward_future)
        
        
    __call__ = forward

    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()
    def __exit__ ( self ):
        self.__del__()


    def test_module(self):
        module = Client(ip='0.0.0.0', port=8091)
        import torch
        data = {
            'bro': torch.ones(10,10),
            'fam': torch.zeros(10,10)
        }

    def virtual(self):
        return c.virtual_client(module = self)


