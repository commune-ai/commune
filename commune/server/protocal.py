

import commune as c

class Protocal(c.Module):

    def __init__(self, 
                module,
                serializer='serializer', 
                max_request_staleness=5, 
                chunk_size=1000, 
                access_module='server.access',
                history_module='server.history',
                history_path='history',
                save_history=False,
                key=None,
                **kwargs
                ):
        self.max_request_staleness = max_request_staleness
        self.chunk_size = chunk_size
        self.key = c.get_key(key)
        self.module = module
        self.access_module = c.module(access_module)(module=self.module)
        self.save_history = save_history
        self.history_module = c.module(history_module)(history_path=history_path)
        self.serializer = c.module(serializer)()
                
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
        assert  c.timestamp() - input.get('timestamp', 0) < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"

        if 'params' in input:
            # if the params are in the input, we want to move them to the data
            if isinstance(input['params'], dict):
                input['kwargs'] = input['params']
            elif isinstance(input['params'], list):
                input['args'] = input['params']
            del input['params']

        
        return {'args': input.get('args', []),
                'kwargs': input.get('kwargs', {}), 
                'address': input.get('address', None),
                'timestamp': input.get('timestamp', c.timestamp())}
    

    def resolve_input_v2(self, input, access_feature='access_ticket'):
        """
        **kwargs: the params
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
    
        access_ticket = input[access_feature]
        access_ticket_dict = {t.split('=')[0]:t.split('=')[1] for t  in access_ticket.split('::')}
        # check the request staleness
        request_staleness = c.timestamp() - access_ticket_dict.get('time', 0)
        assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
        assert c.verify_ticket(access_ticket), f"Data not signed with correct key"
        """
        We assume the data is in the input, and the token
        """
        if 'params' in input:
            # if the params are in the input, we want to move them to the data
            if isinstance(input['params'], dict):
                input['kwargs'] = input['params']
            elif isinstance(input['params'], list):
                input['args'] = input['params']
            del input['params']

        if 'args' and 'kwargs' in input and len(input) == 2:
            input =  {'args': input['args'], 'kwargs': input['kwargs']}
        else:
            input = {'args': [], 'kwargs': input}
        
        input['timestamp'] = c.timestamp()
        input['address'] = access_ticket_dict.get('address', '0x0')
        return input

    def process_input(self,fn, input):
        # you can verify the input with the server key class
        input = self.resolve_input_v1(input)
        input = self.resolve_input_v2(input)
        input['fn'] = fn
        self.check_input(input)
        user_info = self.access_module.verify(fn=input['fn'], address=input['address'])
        assert user_info['success'], f"{user_info}"
        return input
    

    def check_input(self, input):
        assert 'fn' in input, f'fn not in input'
        assert 'args' in input, f'args not in input'
        assert 'kwargs' in input, f'kwargs not in input'
        assert 'address' in input, f'address not in input'
        assert 'timestamp' in input, f'timestamp not in input'


    def process_output(self,  result):
        if c.is_generator(result):
            from sse_starlette.sse import EventSourceResponse
            # for sse we want to wrap the generator in an eventsource response
            result = self.generator_wrapper(result)
            return EventSourceResponse(result)
        else:
            # if we are not using sse, then we can do this with json
            result = self.serializer.serialize(result)
            result = self.key.sign(result, return_json=True)
            return result
        
    
    def generator_wrapper(self, generator):
        """
        This function wraps a generator in a format that the eventsource response can understand
        """

        for item in generator:

            # we wrap the item in a json object, just like the serializer does
            item = self.serializer.serialize({'data': item})
            item_size = len(str(item))
            # we need to add a chunk start and end to the item
            if item_size > self.chunk_size:
                # if the item is too big, we need to chunk it
                chunks =[item[i:i+self.chunk_size] for i in range(0, item_size, self.chunk_size)]
                # we need to yield the chunks in a format that the eventsource response can understand
                for chunk in chunks:
                    yield chunk
            else:
                yield item



    def forward(self, fn:str, input:dict):
        try:
            input = self.process_input(fn, input)   
            fn_obj = getattr(self.module, fn)
            output = fn_obj(*input['args'], **input['kwargs']) if callable(fn_obj) else fn_obj
            output = self.process_output(output)
        except Exception as e:
            output = c.detailed_error(e)
            c.print(output, color='red')

        # process the output
        if self.save_history:
            output = {**input,
                'output': output,
                'latency': c.time() - input['timestamp'],
                'datetime': c.time2datetime(input['timestamp']),
            }
            self.history_module.add_history(output)

        return output