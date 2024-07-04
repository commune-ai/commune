

import commune as c
from typing import *
from sse_starlette.sse import EventSourceResponse
import asyncio
import aiohttp
import json


class Protocal(c.Module):


    def __init__(self, 
                module: Union[c.Module, object] = None,
                name = None,
                max_request_staleness=5, 
                mode = 'server',
                network = 'local',

                port = None,
                key=None,
                loop = None,
                nest_asyncio = False,
                **kwargs
                ):
        if  nest_asyncio:
            c.new_event_loop(nest_asyncio=nest_asyncio)
        self.loop = c.get_event_loop() if loop == None else loop
        self.max_request_staleness = max_request_staleness
        self.serializer = c.module('serializer')()
        self.unique_id_map = {} 
        self.network = network
        self.mode = mode
        
        if self.mode == 'server':
            if isinstance(module, str):
                module = c.module(module)()
            # RESOLVE THE WHITELIST AND BLACKLIST
            module.whitelist = module.get_whitelist()
            module.name = module.server_name = name = name or module.server_name
            module.port = port if port not in ['None', None] else c.free_port()
            module.ip = c.ip()
            module.address = f"{module.ip}:{module.port}"
            module.network = network
            self.module = module
            self.access_module = c.module('protocal.access')(module=self.module)
            self.key  = c.get_key(key or module.name, create_if_not_exists=True)
            module.key = self.key 

        elif self.mode == 'client':
            self.key  = c.get_key(key, create_if_not_exists=True)
            # we dont want to load the namespace if we have the address
            if not c.is_address(module):
                namespace = c.get_namespace(search=module, network=network)
                if module in namespace:
                    module = namespace[module]
                else:    
                    module = module
            print(f"🔑 {self.key.ss58_address} {module}🔑")
            self.module = module

    def client_forward(self,
                       *args,
                       timeout:int=10,
                          **kwargs):
        return self.loop.run_until_complete(asyncio.wait_for(self.async_client_forward(*args, **kwargs), timeout=timeout))

    async def async_client_forward(self, 
                             fn  = 'info', 
                             params = None, 
                             args=None,
                                kwargs=None,
                           timeout:int=10, 
                           stream = False,
                           module = None,
                           key = None,
                           headers = {'Content-Type': 'application/json'},
                           verbose=False, 
                           **extra_kwargs):
        

        if '/' in str(fn):
            module, fn = fn.split('/')
        key = self.resolve_key(key)
        module = module or self.module
        if '/' in module.split('://')[-1]:
            module = module.split('://')[-1]
        url = f"{module}/{fn}/"
        url = 'http://' + url if not url.startswith('http://') else url

        if params != None:
            assert type(params) in [list, dict], f'params must be a list or dict, not {type(params)}'
            if isinstance(params, list):
                args = params
            elif isinstance(params, dict):
                kwargs = params  

        input =  { 
                    "args": args or [],
                    "kwargs": {**(kwargs or {}), **extra_kwargs},
                    "timestamp": c.timestamp(),
                    }
        input = self.serializer.serialize(input)
        input = self.key.sign(input, return_json=True)  

        # start a client session and send the request
        c.print(f"🛰️ Call {url} 🛰️  (🔑{self.key.ss58_address})", color='green', verbose=verbose)
        session = aiohttp.ClientSession()

        try:
            response =  await session.post(url, json=input, headers=headers)
            if response.content_type == 'application/json':
                result = await asyncio.wait_for(response.json(), timeout=timeout)
            elif response.content_type == 'text/plain':
                result = await asyncio.wait_for(response.text(), timeout=timeout)
            elif response.content_type == 'text/event-stream':
                if stream:           
                    return self.stream_generator(response)
                else:
                    result = []  
                    async for line in response.content:
                        event =  self.process_stream_line(line)
                        if event == '':
                            continue
                        result += [event]
                    # process the result if its a json string
                    if isinstance(result, str):
                        if (result.startswith('{') and result.endswith('}')) or result.startswith('[') and result.endswith(']'):
                            result = ''.join(result)
                            result = json.loads(result)
            else:
                raise ValueError(f"Invalid response content type: {response.content_type}")
        except Exception as e:
            return c.detailed_error(e)
        if type(result) in [str, dict, int, float, list, tuple, set, bool, type(None)]:
            result = self.serializer.deserialize(result)
            if isinstance(result, dict) and 'data' in result:
                result = result['data']
        else: 
            result = self.iter_over_async(result)

        
        return result



    def resolve_key(self,key=None):
        if key == None:
            key = self.key
        if isinstance(key, str):
            key = c.get_key(key)
        return key
    
    
    def process_stream_line(self, line ):
        STREAM_PREFIX = 'data: '
        event_data = line.decode('utf-8')
        event_data = event_data[len(STREAM_PREFIX):] if event_data.startswith(STREAM_PREFIX) else event_data
        event_data = event_data.strip() # remove leading and trailing whitespaces
        if event_data == "": # skip empty lines if the event data is empty
            return ''
        if isinstance(event_data, str):
            if event_data.startswith('{') and event_data.endswith('}') and 'data' in event_data:
                event_data = json.loads(event_data)['data']
        return event_data


    def forward(self, fn, input, cache_exception=True):
        if cache_exception:
            try:
                return self.forward(fn, input, cache_exception=False)
            except Exception as e:
                return c.detailed_error(e)
            

        address = input.get('address', None)
        assert c.verify(input), f"Data not signed with correct key {input}"
        input = self.serializer.deserialize(input['data'])
        input['address'] = address
        # check the request staleness    
        request_staleness = c.timestamp() - input.get('timestamp', 0) 
        assert  request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"

        if 'params' in input:
            if isinstance(input['params'], dict):
                input['kwargs'] = input.pop('params')
            elif isinstance(input['params'], list):
                input['args'] = input.pop('params')

        input = {'args': input.get('args', []),
                'kwargs': input.get('kwargs', {}), 
                'address': input.get('address', None),
                'timestamp': input.get('timestamp', c.timestamp())}
        

        input['fn'] = fn
        assert 'fn' in input, f'fn not in input'
        assert 'args' in input, f'args not in input'
        assert 'kwargs' in input, f'kwargs not in input'
        assert 'address' in input, f'address not in input'
        assert 'timestamp' in input, f'timestamp not in input'
        user_info = self.access_module.forward(fn=input['fn'], address=input['address'])
        assert user_info['success'], f"{user_info}"

        fn_obj = getattr(self.module, input['fn'])
        if callable(fn_obj):
            output = fn_obj(*input['args'], **input['kwargs'])
        else:
            output = fn_obj

        if c.is_generator(output):
            def generator_wrapper(generator):
                for item in generator:
                    yield self.serializer.serialize(item)
            return EventSourceResponse(generator_wrapper(output))
        else:
            return self.serializer.serialize(output)


    def iter_over_async(self, ait):
        # helper async fn that just gets the next element
        # from the async iterator
        def get_next():
            try:
                obj = self.loop.run_until_complete(ait.__anext__())
                return obj
            except StopAsyncIteration:
                return 'done'
        # actual sync iterator (implemented using a generator)
        while True:
            obj = get_next() 
            if obj == 'done':
                break
            yield obj

    async def stream_generator(self, response):
        async for line in response.content:
            event =  self.process_stream_line(line)
            if event == '':
                continue
            yield event
