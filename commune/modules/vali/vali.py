# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()

# Validator = c.module('vali')


class Validator(c.Module):

    def __init__(self, config=None,
                 **kwargs):
        self.count = 0
        self.set_config(config=config, kwargs=locals())
        
        self.subspace = c.module(self.config.network)()
        self.modules = self.subspace.modules()
        self.n = len(self.modules)
        self.subspace = c.module(self.config.network)()
         

    async def eval_module(self, module=None, verbose:bool=True):
        module_state = c.choice(self.modules) if module == None else None
        module_name = module_state['name']
        module = await c.async_connect(module_state['address'], 
                                        network=self.config.network, 
                                        timeout=self.config.timeout)

        response = {'module': module_name, 'w': w, 'error': error}
        # if verbose:
        #     c.print(response, color='white')
            
        self.count += 1
        w = response['w']
        # we want to mix more recent scores with older ones
        w = w * self.config.alpha + (1-self.config.alpha) * self.w.get(module_name, 0)
        self.w[module_name] = w
        if self.count % self.config.save_interval == 0:
            self.save()
        if module_name not in module_state:
            module_state[module_name] = []
        module_state[module_name].append(response)
        module_id = module_state['uid']
        self.modules[module_id] = module_state

    def save(self, tag=None):
        
        
        tag = self.config.tag if tag == None else tag
            
        self.put(f'{tag}/w', self.w)

    def run(self, parallel_jobs:int=10):
        

        self.running = True
        
        
        while self.running:
            jobs = [self.async_eval_module() for _ in range(parallel_jobs)]
            c.gather(jobs)

           
    def check_score(self, module):
        module_name = module['name']
        loop = c.get_event_loop()
        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        
    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)
        