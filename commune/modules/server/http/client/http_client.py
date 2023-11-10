

from typing import Tuple, List, Union
import sys
import os
import asyncio
import requests
from functools import partial
import commune as c
import aiohttp
import json


from aiohttp.streams import StreamReader

# Define a custom StreamReader with a higher limit
class CustomStreamReader(StreamReader):
    def __init__(self, *args, **kwargs):
        # You can adjust the limit here to a value that fits your needs
        # This example sets it to 1MB
        super().__init__(*args, limit=1024*1024, **kwargs)


class Client(c.Module):

    def __init__( 
            self,
            ip: str ='0.0.0.0',
            port: int = 50053 ,
            network: bool = None,
            key : str = None,
            loop: 'asyncio.EventLoop' = None
        ):
        self.loop = c.get_event_loop() if loop == None else loop
        self.set_client(ip =ip,port = port)
        self.serializer = c.serializer()
        self.key = c.get_key(key)
        self.my_ip = c.ip()
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
        fn: str,
        args: list = None,
        kwargs: dict = None,
        ip: str = None,
        port : int= None,
        timeout: int = 10,
        headers : dict ={'Content-Type': 'application/json'}):

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

        # serialize this into a json string
        request_data = self.serializer.serialize( request_data)

        # sign the request
        request = self.key.sign(request_data, return_json=True)

        result = '{}'


        # start a client session and send the request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request, headers=headers) as response:
                if response.content_type == 'text/event-stream':
                    # Process SSE events
                    result = ''
                    async for line in response.content:
                        # remove the "data: " prefix
                        event_data = line.decode('utf-8').strip().replace("data: {", "{")
                        if event_data == "":
                            continue
                        if isinstance(event_data, str):
                            result += event_data

                        
                    result = self.process_output(json.loads(result))
                    

                elif response.content_type == 'application/json':
                    result = await asyncio.wait_for(response.json(), timeout=timeout)
                    result = self.process_output(result)
                elif response.content_type == 'text/plain':
                    # result = await asyncio.wait_for(response.text, timeout=timeout)
                    result = json.loads(result)
                    result = self.process_output(result)
                else:
                    raise ValueError(f"Invalid response content type: {response.content_type}")
        # process output 
        

        return result


    def process_output(self, result):
        ## handles 
        if isinstance(result, str):
            result = json.loads(result)
        result = self.serializer.deserialize(result['data'])

        

        return result['data']
        
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


