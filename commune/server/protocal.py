

import commune as c
from typing import *
from sse_starlette.sse import EventSourceResponse

import commune as c
from typing import *


class Protocal(c.Module):


    def __init__(self, 
                module: Union[c.Module, object] = None,
                name = None,
                max_request_staleness=5, 
                serializer = 'serializer', 
                access_module='server.access',
                history_module='server.history',
                ticket_module = 'ticket',
                history_path='history',
                network = 'local',
                save_history=False,
                mnemonic=None,
                key=None,
                **kwargs
                ):

        self.max_request_staleness = max_request_staleness
        self.set_module(module=module, key=key, name=name, mnemonic=mnemonic, network=network, **kwargs)
        self.ticket_module = c.module(ticket_module)()
        self.access_module = c.module(access_module)(module=self.module)
        self.save_history = save_history
        self.history_module = c.module(history_module)(history_path=history_path)
        self.serializer = c.module(serializer)()
        self.unique_id_map = {}

    def set_module(self, module, key, name, port:int= None, network:str='local', mnemonic=None, **kwargs):
       
       
        if mnemonic != None:
            c.add_key(name, mnemonic)
        module = module or 'module'
        if isinstance(module, str):
            module = c.module(module)()
        # RESOLVE THE WHITELIST AND BLACKLIST
        module.whitelist = list(set((module.whitelist if hasattr(module, 'whitelist') else [] ) + c.whitelist))
        module.blacklist = list(set((module.blacklist if hasattr(module, 'blacklist') else []) + c.blacklist))
        module.name = module.server_name = name = name or module.server_name
        module.port = port if port not in ['None', None] else c.free_port()
        module.ip = c.ip()
        module.address = f"{module.ip}:{module.port}"
        module.network = network
        module.key = c.get_key(key or module.name, create_if_not_exists=True)
        self.module = module

    def resolve_input_v1(self, input):
        """
        {
            data: {
                'args': [],
                'kwargs': {},
                'timestamp': 'UTC timestamp',
            },
            'signature': 'signature of address',
            'address': 'address of the caller',
        }
        
        """

        is_input_v1 = bool('signature' in input and 'data' in input)

        if not is_input_v1:
            return input
        
        assert c.verify(input), f"Data not signed with correct key"
        address = input['address']
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

        return {'args': input.get('args', []),
                'kwargs': input.get('kwargs', {}), 
                'address': input.get('address', None),
                'timestamp': input.get('timestamp', c.timestamp())}
    

    def resolve_input_v2(self, input, access_feature='access_ticket'):
        """
        **kwargs: the params
        0.0.0.0:8888/{fn}/{'text': 'hello', 'access_token': 'token'}
        or args=[] and kwargs={} as the input
        params can be used in place for args and kwargs
        module_tikcet: time={time}::address={address}::signature={signature}
        access_token:
        - the access token is a string that is signed with the server's key
        - the access token is in the format time={time:INT}::address={address:STR}::signature={signature:STR}
        - the access token is used to verify the user's access to the server
        """
        is_input_v2 = bool(access_feature in input)
        if not is_input_v2:
            return input
        
        access_ticket = input.pop(access_feature, None)
        assert self.ticket_module.verify(access_ticket), f"Data not signed with correct key"
        access_ticket_dict = self.ticket_module.ticket2dict(access_ticket)
        # check the request staleness
        request_staleness = c.timestamp() - access_ticket_dict.get('timestamp', 0)
        assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
        assert c.verify_ticket(access_ticket), f"Data not signed with correct key"
        """
        We assume the data is in the input, and the token
        """
        if 'params' in input:
            if isinstance(input['params'], dict):
                input['kwargs'] = input.pop('params')
            elif isinstance(input['params'], list):
                input['args'] = input.pop('params')

        if 'args' in input or 'kwargs' in input:
            input =  {'args': input.get('args', []), 'kwargs': input.get('kwargs', {})}
        input['timestamp'] = c.timestamp()
        input['address'] = access_ticket_dict.get('address', '0x0')
        return input

    def process_input(self, fn, input):
        # you can verify the input with the server key class
        input = self.resolve_input_v1(input)
        input = self.resolve_input_v2(input)
        input['fn'] = fn
        self.check_input(input)
        user_info = self.access_module.forward(fn=input['fn'], address=input['address'])
        assert user_info['success'], f"{user_info}"
        return input
    

    def check_input(self, input):
        assert 'fn' in input, f'fn not in input'
        assert 'args' in input, f'args not in input'
        assert 'kwargs' in input, f'kwargs not in input'
        assert 'address' in input, f'address not in input'
        assert 'timestamp' in input, f'timestamp not in input'

        # check if the request has already been processed
        # unique_id = 'address='+input['address'] + '::timestamp='+ str(input['timestamp'])
        # assert unique_id not in self.unique_id_map, f"Request already processed"
        # self.unique_id_map[unique_id] = True

    def process_output(self,  result):
        if c.is_generator(result):
            def generator_wrapper(generator):
                for item in generator:
                    yield self.serializer.serialize(item)
            return EventSourceResponse(generator_wrapper(result))
        else:
            return self.serializer.serialize(result)

    def forward(self, fn:str, input:dict):
        try:
            input = self.process_input(fn, input)   
            fn_obj = getattr(self.module, fn)
            output = fn_obj(*input['args'], **input['kwargs']) if callable(fn_obj) else fn_obj
            c.print(input)
            # process the output
            if self.save_history:
                self.history_module.add_history({**input,
                    'output': output,
                    'latency': c.time() - input['timestamp'],
                    'datetime': c.time2datetime(input['timestamp']),
                })
            output = self.process_output(output)
        except Exception as e:
            output = c.detailed_error(e)
            c.print(output, color='red')


        return output
    

