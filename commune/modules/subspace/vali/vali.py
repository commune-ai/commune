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
        self.state['response_history'] = []
        self.module_info = {}
        self.w = {}
        if run:
            self.run()
        
        
        self.running = False
        
        
    def get_score(self, module):

        return 1
    
         
    def eval_module(self, module=None, verbose:bool=True):
        module = c.choice(self.modules) if module == None else None
        w = 1
        error = None
        try:
            module_name = module['name']
            # get connection
            module = c.connect(module['address'], network=self.config.network, timeout=self.config.timeout)
            
            # get info
            if module_name not in self.module_info:
                self.module_info[module_name] = module.info(timeout=self.config.timeout)
        except Exception as e:
            # something went wrong, set score to 0, 
            w = 0
            response = {'error': str(e), 'module': module_name, 'w': 0}

        w = self.get_score(module) if w != 0 else 0

        response = {'module': module_name, 'w': w, 'error': error}
        
        if verbose:
            c.print(response, color='white')
            
        self.register_response(response)
            
       
    def register_response(self, response:dict):
        w = response['w']
        # we want to mix more recent scores with older ones
        w = w * self.config.alpha + (1-self.config.alpha) * self.w.get(module_name, 0)
        self.w[module_name] = w
        self.put('w', self.w)
        self.state['response_history'].append(response)
        self.state['w'] = self.w
    

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
    def test(cls):
        self = cls()
        