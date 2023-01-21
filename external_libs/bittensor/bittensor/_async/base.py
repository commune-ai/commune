
import asyncio


class AsyncBase:

    def set_event_loop(self, loop=None):
        if loop == None:
            loop = asyncio.get_event_loop()
        self.loop = loop
        return self.loop

    @staticmethod
    def get_event_loop(*args,**kwargs):
        return asyncio.get_event_loop(*args,**kwargs)
         

    @staticmethod
    def new_event_loop(*args,**kwargs):
        return asyncio.new_event_loop(*args,**kwargs)
          
    
    def async_run(self, job, loop=None): 
        if loop == None:
            loop = self.loop
        return self.loop.run_until_complete(job)


