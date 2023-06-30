# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()



class Validator(c.Module):

    def __init__(self, config=None,
                 **kwargs):
        self.set_config(config=config, kwargs=locals())
        self.subspace = c.module(self.config.network)()
        self.modules = self.subspace.modules()
        if self.config.load == True:
            self.load(self.config.tag)
        if self.config.run:
            self.run()
        
        
        self.running = False
        
        
    def get_score(self, module):

        return 1
    
         
    def eval_module(self, module=None, verbose:bool=True):
        module_state = c.choice(self.modules) if module == None else None
        w = 1
        error = None
        try:
            module_name = module_state['name']
            # get connection
            module = c.connect(module_state['address'], network=self.config.network, timeout=self.config.timeout)
            
            # get info
            if module_name not in module_state:
                module_state = module.info(timeout=self.config.timeout)
        except Exception as e:
            # something went wrong, set score to 0, 
            w = 0
            response = {'error': str(e), 'module': module_name, 'w': 0}

        w = self.get_score(module) if w != 0 else 0

        response = {'module': module_name, 'w': w, 'error': error}
        if verbose:
            c.print(response, color='white')
            
        self.count += 1
        w = response['w']
        # we want to mix more recent scores with older ones
        w = w * self.config.alpha + (1-self.config.alpha) * self.w.get(module_name, 0)
        self.w[module_name] = w
        if self.count % self.config.save_interval = 0:
            self.save()
        if module_name not in module_state:
            module_state[module_name] = []
        module_state[module_name].append(response)
        module_id = module_state['uid']
        self.modules[module_id] = module_state
        c.print(f'w: {w}')

    def save(self, tag=None):
        
        
        tag = self.config.tag if tag == None else tag
        for m in self.modules:
            
            
        self.put(f'{tag}/w', self.w)
        self.put(f'{tag}/response_history', self.response_history)
        self.put(f'{tag}/network_state', self.response_history)

    def run(self):
        

        self.running = True
        while self.running:
            self.eval_module()

           
    def check_score(self, module):
        module_name = module['name']
        loop = c.get_event_loop()
        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        
    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)
        