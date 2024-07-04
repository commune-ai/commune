

from typing import *
import asyncio
import commune as c
import aiohttp
import json
from .pool import ClientPool
# from .pool import ClientPool

class Client(c.Module, ClientPool):
    def __init__( 
            self,
            module : str = '0.0.0.0:8000',
            network: bool = 'local',
            protocal = 'protocal',
            key = None,
            **kwargs
        ):

        self.protocal = c.module(protocal)(module=module, 
                                           network=network, 
                                           key=key,
                                           mode = 'client',
                                            **kwargs)
        
        print(f"🔑 {self.protocal.key.ss58_address} {self.protocal.module}🔑")
        self.forward = self.protocal.client_forward
        self.async_forward = self.protocal.async_client_forward


    @classmethod
    def call(cls, 
                module : str, 
                fn:str = 'info',
                *args,
                kwargs = None,
                params = None,
                prefix_match:bool = False,
                network:str = 'local',
                key:str = None,
                stream = False,
                timeout=40,
                **extra_kwargs) -> None:
          
        # if '
        if '//' in module:
            module = module.split('//')[-1]
            mode = module.split('//')[0]
        if '/' in module:
            if fn != None:
                args = [fn] + list(args)
            module , fn = module.split('/')

        module = cls.connect(module=module,
                           network=network,  
                           prefix_match=prefix_match, 
                           virtual=False, 
                           key=key)

        if params != None:
            kwargs = params

        if kwargs == None:
            kwargs = {}

        kwargs.update(extra_kwargs)

        return  module.forward(fn=fn, args=args, kwargs=kwargs, stream=stream, timeout=timeout)

    @classmethod
    def connect(cls,
                module:str, 
                network : str = 'local',
                mode = 'http',
                virtual:bool = True, 
                **kwargs):
        
        client = cls(module=module, 
                                       virtual=virtual, 
                                       network=network,
                                       **kwargs)
        # if virtual turn client into a virtual client, making it act like if the server was local
        if virtual:
            return client.virtual()
        
        return client
    
    def test(self, module='module::test_client'):
        c.serve(module)
        c.sleep(1)
        c.print(c.server_exists(module))
        c.print('Module started')

        info = c.call(module+'/info')
        key  = c.get_key(module)
        assert info['key'] == key.ss58_address
        return {'info': info, 'key': str(key)}



    def __del__(self):
        if hasattr(self, 'session'):
            asyncio.run(self.protocal.session.close())


    def __str__ ( self ):
        return "Client({})".format(self.address) 
    def __repr__ ( self ):
        return self.__str__()


    def virtual(self):
        from .virtual import VirtualClient
        return VirtualClient(module = self)
    
    def __repr__(self) -> str:
        return super().__repr__()
    
    