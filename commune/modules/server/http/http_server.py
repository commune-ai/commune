
from typing import Dict, List, Optional, Union
import commune as c



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
        key = None,
    ) -> 'Server':
        
        if isinstance(module, str):
            module = c.module(module)()
        elif isinstance(module, type):
            module = module()

        if name == None:
            name = module.name()

        self.name = name
        self.timeout = timeout
        self.verbose = verbose
        self.serializer = c.module('server.http.serializer')()
        self.ip = c.resolve_ip(ip, external=False)  # default to '0.0.0.0'
        self.port = c.resolve_port(port)
        self.address = f"{self.ip}:{self.port}"
        self.key = c.get_key(name) if key == None else key
        self.whitelist = getattr( module, 'whitelist', []) if whitelist == None else whitelist
        self.blacklist = getattr( module, 'blacklist', []) if blacklist == None else blacklist
        # ensure that the module has a name
        for k in ['module_name', 'module_id', 'name']:
            if k not in self.__dict__:
                self.__dict__[k] = name
        # register the server
        module.ip = self.ip
        module.port = self.port
        module.address  = self.address
        if (not hasattr(self, 'config')) or callable(self.config):
            module.config = cls.config()
        module.config['info'] = module.info()
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

    
    def verify(self, data: dict) -> bool:
        r""" Verify the data is signed with the correct key.
        """
        assert isinstance(data, dict), f"Data must be a dict, not {type(data)}"
        assert 'data' in data, f"Data not included"
        assert 'signature' in data, f"Data not signed"
        assert self.key.verify(data), f"Data not signed with correct key"
        return True

    def set_api(self):

        from fastapi import FastAPI

        self.app = FastAPI()


        @self.app.post("/{fn}/")
        async def forward_wrapper(fn:str, input:dict[str, str]):
            # verify key
            self.verify(input)

            # deserialize data
            data = self.serializer.deserialize(input.pop('data'))
            
            # forward
            result =  self.forward(fn=fn, 
                                    args=data.get('args', []), 
                                    kwargs=data.get('kwargs', {})
                                    )
            # serialize result
            result_data = self.serializer.serialize(result)

            # sign result data (str)
            result =  self.key.sign(result_data, return_json=True)

            # send result to client
            return result
        
        c.register_server(self.name, self.ip, self.port)

    def serve(self, **kwargs):
        import uvicorn
        uvicorn.run(self.app, host=self.ip, port=self.port)

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