
from typing import Dict, List, Optional, Union
import commune as c
import torch 


class HTTPServer(c.Module):
    def __init__(
        self,
        module: Union[c.Module, object],
        name: str = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = 4,
        verbose: bool = True,
        whitelist: List[str] = None,
        blacklist: List[str] = None,
        save_history_interval: int = 100,
        max_request_staleness: int = 100,
        key = None,
    ) -> 'Server':
        

        self.timeout = timeout
        self.verbose = verbose
        self.serializer = c.module('server.http.serializer')()
        self.ip = c.resolve_ip(ip, external=False)  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"{self.ip}:{self.port}"
        self.key = c.get_key(name) if key == None else key
        self.whitelist = getattr( module, 'whitelist', []) if whitelist == None else whitelist
        self.blacklist = getattr( module, 'blacklist', []) if blacklist == None else blacklist
        self.save_history_interval = save_history_interval
        self.max_request_staleness = max_request_staleness
        self.history = []
        # ensure that the module has a name

        if isinstance(module, str):
            module = c.module(module)()
        elif isinstance(module, type):
            module = module()

        if name == None:
            name = module.name()

        self.name = name
        for k in ['module_name', 'module_id', 'name']:
            if k not in module.__dict__:
                module.__dict__[k] = name
        # register the server
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        self.module = module
        self.set_api()
        self.serve()



    def state_dict(self) -> Dict:
        return {
            'ip': self.ip,
            'port': self.port,
            'address': self.address,
            'timeout': self.timeout,
            'verbose': self.verbose,
        }


    def test(self):
        r"""Test the HTTP server.
        """
        # Test the server here if needed
        c.print(self.state_dict(), color='green')
        return self

    
    def verify_input(self, input: dict) -> bool:
        r""" Verify the data is signed with the correct key.
        """
        try:
            assert isinstance(input, dict), f"Data must be a dict, not {type(data)}"
            assert 'data' in input, f"Data not included"
            assert 'signature' in input, f"Data not signed"
            assert self.key.verify(input), f"Data not signed with correct key"

            # deserialize data
            data = self.serializer.deserialize(input.pop('data'))
            request_timestamp = data.get('timestamp', 0)
            request_staleness = c.timestamp() - request_timestamp
            assert request_staleness < self.max_request_staleness, f"Request is too old, {request_staleness} > MAX_STALENESS ({self.max_request_staleness})  seconds old"
            return data
        except Exception as e:
            c.print(e, color='red')
            return {'error': str(e)}

    def set_api(self):

        from fastapi import FastAPI

        self.app = FastAPI()


        @self.app.post("/{fn}/")
        async def forward_wrapper(fn:str, input:dict[str, str]):
            # verify data
            data = self.verify_input(input)
            if 'error' in data:
                return data
            # forward
            result =  self.forward(fn=fn, 
                                    args=data.get('args', []), 
                                    kwargs=data.get('kwargs', {})
                                    )
            
            # serialize result
            result_data = self.serializer.serialize(result)

            # sign result data (str)
            result =  self.key.sign(result_data, return_json=True)


            self.history.append({
                'fn': fn,
                'ip': data.pop('ip', None),
                'address': input.pop('address', None),
                'timestamp': data.pop('timestamp', None),
            }
            )

            if len(self.history) > 100:
                self.history = self.history[-100:]


            # send result to client
            return result
        
        c.register_server(self.name, self.ip, self.port)


    def save(self, data: dict):
        r"""Save the history of the server.
        """
        og_history = self.get(f'history/{self.name}', [])
        og_history.extend(self.history)
    def serve(self, **kwargs):
        import uvicorn
        try:
            uvicorn.run(self.app, host=self.ip, port=self.port)
        except Exception as e:
            c.deregister_server(self.name)
        finally:
            c.deregister_server(self.name)
        

    def forward(self, fn: str, args: List = None, kwargs: Dict = None, **extra_kwargs):
        try: 
            if args is None:
                args = []
            if kwargs is None:
                kwargs = {}
            obj = getattr(self.module, fn)
            if callable(obj):
                response = obj(*args, **kwargs)
            else:
                response = obj
            return response
        except Exception as e:
            response = {'error': str(e)}
            c.print(e, color='red')
        return response


    def __del__(self):
        c.deregister_server(self.name)