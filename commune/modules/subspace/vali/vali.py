# import nest_asyncio
# nest_asyncio.apply()
import commune as c
c.new_event_loop()



class Validator(c.Module):

    def __init__(self, config=None,
                 run:bool = False,
                 **kwargs):
        self.set_config(config=config, kwargs=locals())
        self.subspace = c.module('subspace')()
        self.state = self.subspace.state_dict()
        self.modules = self.state['modules']
        self.namespace = self.state['namespace']
        self.state = self.config.get('state', self.state)
        self.module_info = {}
        self.w = {}
        if run:
            self.run()
        
        
        self.running = False
        
        
    def get_score(self, module):
        w = 1

        return w

    
    def run(self):
        self.running = True
        while self.running:
            module = c.choice(self.modules)
            error = None
            w = 1
            try:
                
                module_name = module['name']
                # get connection
                module = c.connect(module['address'], network=self.config.network, timeout=self.config.timeout)
                
                # get info
                if module_name not in self.module_info:
                    self.module_info[module_name] = module.info(timeout=self.config.timeout)
            except Exception as e:
                error = str(e)
                w = 0
            if w != 0:
                w = self.get_score(module)

            self.w[module_name] = w * self.config.decay + (1 - self.config.decay) * self.w.get(module_name, 0)
            self.put('w', self.w)
            c.sleep(self.config.sleep)
           
    def check_score(self, module):
        module_name = module['name']
        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        
    @classmethod
    def test(cls):
        self = cls()
        