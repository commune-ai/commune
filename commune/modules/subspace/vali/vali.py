# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()



class Validator(c.Module):

    def __init__(self, config=None,
                 **kwargs):
        self.count = 0
        self.w = {}
        self.set_config(config=config, kwargs=locals())
        self.subspace = c.module(self.config.network)()
        self.modules = self.subspace.modules()
        if self.config.load == True:
            self.load(self.config.tag)
        if self.config.run:
            self.run()
        
        
        self.running = False
        
    def calculate_score(self, module):

        return 1

    
         
    async def async_eval_module(self, module=None, verbose:bool=True):
        module_state = c.choice(self.modules) if module == None else None
        w = 1
        error = None
        try:
            module_name = module_state['name']
            # get connection
            module = await c.async_connect(module_state['address'], 
                                           network=self.config.network, 
                                           timeout=self.config.timeout)
            
            # get info
            # if 'info' not in module_state:
            module_state['info'] = module.info(timeout=self.config.timeout)
        except Exception as e:
            # something went wrong, set score to 0, 
            w = 0
            error = str(e)

        w = self.calculate_score(module) if w != 0 else 0

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
        