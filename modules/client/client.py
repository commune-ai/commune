

from typing import Tuple, List, Union
import asyncio
from functools import partial
import commune as c
import aiohttp
import json

class Client(c.Module):

    def __init__( 
            self,
            ip: str ='0.0.0.0',
            port: int = 50053 ,
            network: bool = None,
            key : str = None,
            save_history: bool = True,
            history_path : str = 'history',
            loop: 'asyncio.EventLoop' = None, 
            debug: bool = False,
            serializer= 'serializer',
            **kwargs
        ):
        self.loop = c.get_event_loop() if loop == None else loop
        self.set_client(ip =ip,port = port)
        self.serializer = c.module(serializer)()
        self.key = c.get_key(key)
        self.my_ip = c.ip()
        self.network = c.resolve_network(network)
        self.start_timestamp = c.timestamp()
        self.save_history = save_history
        self.history_path = history_path
        self.debug = debug

        

    
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
        generator: bool = False,
        headers : dict ={'Content-Type': 'application/json'},
        ):
        '''
        # Documentation for `async_forward` function
        
        ## Overview
        
        The `async_forward` function is an asynchronous method designed to forward a request to a specified server function (fn) and handle the server's response. It supports forwarding of both regular and generator-based requests and handles different content types including JSON, event streams, and plain text.
        
        ## Parameters
        
        - `fn`: A string representing the name of the server function to which the request will be forwarded.
        - `args`: An optional list of arguments to be passed to the server function.
        - `kwargs`: An optional dictionary of keyword arguments to be passed to the server function.
        - `ip`: An optional string representing the IP address of the server. If provided, it will override the default server IP address.
        - `port`: An optional integer representing the port number of the server. If provided, it will override the default server port.
        - `timeout`: An optional integer representing the timeout duration in seconds for the request. Default is 10 seconds.
        - `generator`: An optional boolean flag indicating whether the request expects a generator response. Default is False.
        - `headers`: An optional dictionary representing the HTTP headers to be sent with the request. Default is `{'Content-Type': 'application/json'}`.
        
        ## Functionality
        
        1. The function first resolves the client using the provided `ip` and `port`.
        2. It then constructs the request URL and payload, signing the request if needed.
        3. An asynchronous client session is initiated, and the request is posted to the server.
        4. The function handles the server's response based on its content type:
           - For text/event-stream: Processes the event stream and returns the accumulated result.
           - For application/json: Processes and returns the JSON response.
           - For text/plain: Processes and returns the plain text response.
        5. The result is deserialized if necessary, and if `save_history` is True, the request details are saved to a history path.
        6. Finally, the result is returned.
        
        ## Return Value
        
        The function returns the processed server response as a deserialized object (usually a dictionary or a string), depending on the content type of the server's response.
        
        ## Exceptions
        
        - Raises a `ValueError` if the response content type is not one of the supported types.
        - May raise other exceptions related to network errors, timeouts, or deserialization errors.
        '''
        self.resolve_client(ip=ip, port=port)
        args = args if args else []
        kwargs = kwargs if kwargs else {}
        url = f"http://{self.address}/{fn}/"
        input =  { 
                        "args": args,
                        "kwargs": kwargs,
                        "ip": self.my_ip,
                        "timestamp": c.timestamp(),
                        }
        # serialize this into a json string
        request = self.serializer.serialize(input)
        request = self.key.sign(request, return_json=True)

        
        
        # start a client session and send the request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request, headers=headers) as response:
                if response.content_type == 'text/event-stream':
                    STREAM_PREFIX = 'data: '
                    BYTES_PER_MB = 1e6
                    if self.debug:
                        progress_bar = c.tqdm(desc='MB per Second', position=0)

                    result = {}
                    
                    async for line in response.content:
                        event_data = line.decode('utf-8')
                        
                        event_bytes  = len(event_data)
                        if self.debug :
                            progress_bar.update(event_bytes/(BYTES_PER_MB))
                        # remove the "data: " prefix
                        if event_data.startswith(STREAM_PREFIX):
                            event_data = event_data[len(STREAM_PREFIX):]

                        event_data = event_data.strip()
                        
                        # skip empty lines
                        if event_data == "":
                            continue

                        # if the data is formatted as a json string, load it {data: ...}
                        if isinstance(event_data, bytes):
                            event_data = event_data.decode('utf-8')

                        # if the data is formatted as a json string, load it {data: ...}
                        if isinstance(event_data, str):
                            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                                event_data = json.loads(event_data)['data']
                            result += [event_data]
                        
                    # process the result if its a json string
                    if result.startswith('{') and result.endswith('}') or \
                        result.startswith('[') and result.endswith(']'):
                        result = ''.join(result)
                        result = json.loads(result)

                elif response.content_type == 'application/json':
                    # PROCESS JSON EVENTS
                    result = await asyncio.wait_for(response.json(), timeout=timeout)
                elif response.content_type == 'text/plain':
                    # PROCESS TEXT EVENTS
                    result = await asyncio.wait_for(response.text(), timeout=timeout)
                else:
                    raise ValueError(f"Invalid response content type: {response.content_type}")
        if isinstance(result, dict):
            result = self.serializer.deserialize(result)
        elif isinstance(result, str):
            result = self.serializer.deserialize(result)
        if isinstance(result, dict) and 'data' in result:
            result = result['data']
        if self.save_history:
            input['fn'] = fn
            input['result'] = result
            input['module']  = self.address
            input['latency'] =  c.time() - input['timestamp']
            path = self.history_path+'/' + self.server_name + '/' + str(input['timestamp'])
            self.put(path, input)
        return result
    
    @classmethod
    def history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.ls(history_path + '/' + key.ss58_address)
    @classmethod
    def all_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.glob(history_path)
        


    @classmethod
    def rm_key_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.rm(history_path + '/' + key.ss58_address)
    
    @classmethod
    def rm_history(cls, key=None, history_path='history'):
        key = c.get_key(key)
        return cls.rm(history_path)


    def process_output(self, result):
        ## handles 
        if isinstance(result, str):
            result = json.loads(result)
        if 'data' in result:
            result = self.serializer.deserialize(result)
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


    def virtual(self):
        from .virtual import VirtualClient
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()
