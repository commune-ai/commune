
import commune as c
from functools import partial
import asyncio


class VirtualClient:
    protected_attributes = [ 'module_client', 'remote_call']
    def __init__(self, module: str ='ReactAgentModule'):
        if isinstance(module, str):
            self.module_client = c.connect(module)
            self.loop = self.module_client.loop
            self.success = self.module_client.success
        else:
            self.module_client = module

    def remote_call(self, *args, return_future= False, timeout:int=10, **kwargs):
        remote_fn = kwargs.pop('remote_fn')
        
        result =  self.module_client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, return_future=return_future)
        return result
            
    def __str__(self):
        return f'<VirtualClient({self.module_client.address})>'

    def __repr__(self):
        return self.__str__()
        
    def __getattr__(self, key):

        if key in self.protected_attributes :
            return getattr(self, key)
        else:
            return lambda *args, **kwargs : self.remote_call( remote_fn=key, *args, **kwargs)
        



