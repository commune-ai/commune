# import nest_asyncio
# nest_asyncio.apply()
import commune as c

import threading 


class Validator(c.Module):

    def __init__(self, config=None,  **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        self.set_subspace( )
        self.count = 0
        self.stats = {}
        self.start_time = c.time()
        self.start_runs()

    def start_runs(self):
        self.threads = []
        # start threads, ensure they are daemons, and dont vote
        for t in range(self.config.num_threads):
            t = threading.Thread(target=self.run, kwargs={'vote':False})
            t.daemon = True
            t.start()
            self.threads.append(t)

        # main thread
        self.run(vote=True)
        


    def set_subspace(self):
        self.subspace = c.module(self.config.network)()
        self.modules = self.subspace.modules()
        self.namespace = {v['name']: v['address'] for v in self.modules }
        self.name2module = {v['name']: v for v in self.modules }
        self.module_names = list(self.name2module.keys())
    
        self.key = c.get_key(self.config.key)
        self.subspace.is_registered(self.key)

    @property
    def lifetime(self):
        return c.time() - self.start_time

    def modules_per_second(self):
        return self.count / self.lifetime()


    def score_response(self, r) -> int:
        try:
            assert isinstance(r, dict), f'Expected dict, got {type(r)}'
            return 1
        except Exception as e:
            return 0
        
         
    async def async_eval_module(self, module = None, fn='info', args = None, kwargs=None, ):
        if  kwargs == None:
            kwargs = {}
        if args == None:
            args = []

        if module != None:
            module = c.choice(self.module_names)
        module_state = self.name2module[module]
        w = 1
        try:
            # get connection
            module_client = await c.async_connect(module_state['address'], network=self.config.network, namespace = self.namespace,timeout=1)
            response = await getattr(module_client,fn)(timeout=self.config.timeout, return_future=True)
            w = self.score_response(response)
              
        except Exception as e:
            response = {'error': str(e)}
            w = 0

        module_stats = self.stats.get(module, module_state)
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*self.alpha + w(1-self.alpha)

        module_stats['alpha'] = self.config.alpha
        module_stats['history'] = module_stats.get('history', []) + [{'input': dict(args=args, kwargs=kwargs) ,'output': response, 'w': w, 'time': c.time()}]
        self.stats[module] = module_stats
        self.count += 1
        if self.config.save_interval % self.count == 0:
            self.save()
        return module
    def save(self, tag=None):
        c.print(f'Saving stats to {tag}', color='white')
        tag = self.config.tag if tag == None else tag

        for k in self.stats:
            self.put(f'{tag}/stats/{k}', self.stats)



    def run(self, vote = True):
        

        self.running = True

        c.get_event_loop()
        
        
        while self.running:
            c.print(f'Validator: {self.count}', color='white')
            try:
                c.gather([self.async_eval_module() for i in range(self.config.parallel_jobs)], timeout=2)
            except Exception as e:
                c.print(e)

           
    def check_score(self, module):
        module_name = module['name']
        loop = c.get_event_loop()
        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=2)
        
    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)


    def __del__(self):
        self.stop()
        self.save()
        c.print('Validator stopped', color='white')


    # def fleet(cls, n=1, **kwargs):
    #     return [cls(**kwargs) for i in range(n)]