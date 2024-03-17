import asyncio
import commune as c
   
class AsyncModule(c.Module):
    def sync_the_async(self, loop):
        for f in dir(self):
            if 'async_' in f:
                setattr(self, f.replace('async_',  ''), self.sync_wrapper(getattr(self, f), loop=loop))

    def sync_wrapper(self,fn:'asyncio.callable', loop = None) -> 'callable':
        '''
        Convert Async funciton to Sync.

        Args:
            fn (callable): 
                An asyncio function.

        Returns: 
            wrapper_fn (callable):
                Synchronous version of asyncio function.
        '''
        loop = loop if loop else self.loop
        def wrapper_fn(*args, **kwargs):
            return self.loop.run_until_complete(fn(*args, **kwargs))
        return  wrapper_fn
