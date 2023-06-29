# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()



class Validator(c.Module):

    def __init__(self, config=None, **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        
        c.print('Validator config:', self.config)
        
        self.subspace = c.module('subspace')()
        self.state = self.subspace.state_dict()
        c.print('Validator state:', self.state)
        