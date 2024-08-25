import commune as c
from functools import partial
import asyncio


class ClientVirtual:
    protected_attributes = [ 'client', 'remote_call']
    def __init__(self, client: str ='ReactAgentModule'):
        if isinstance(client, str):
            client = c.connect(client)
        self.client = client

    def remote_call(self, remote_fn, *args, return_future= False, timeout:int=10, key=None, **kwargs):
        result =  self.client.forward(fn=remote_fn, args=args, kwargs=kwargs, timeout=timeout, key=key, return_future=return_future)
        return result
            
    def __str__(self):
        return str(self.client)

    def __repr__(self):
        return self.__str__()
        
    def __getattr__(self, key):

        if key in self.protected_attributes :
            return getattr(self, key)
        else:
            return lambda *args, **kwargs : self.remote_call( remote_fn=key, *args, **kwargs)
        