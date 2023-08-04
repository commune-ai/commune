# import nest_asyncio
# nest_asyncio.apply()
import commune as c

import threading 


class Validator(c.Module):

    def __init__(self, config=None,  **kwargs):
        self.set_config(config=config, kwargs=kwargs)

        self.set_subspace( )
        self.stats = {}
        self.start_time = c.time()
        self.count = 0
        self.threads = []
        
        if self.config.start:
            self.start()

    def start(self):
        # start threads, ensure they are daemons, and dont vote
        for t in range(self.config.threads):
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
        return self.count / self.lifetime


    def score_response(self, r) -> int:
        try:
            assert isinstance(r, dict), f'Expected dict, got {type(r)}'
            return 1
        except Exception as e:
            return 0
        
    def eval_module(self, module = None, fn='info', args = None, kwargs=None, ):
        return c.gather(self.async_eval_module(module=module, fn=fn, args=args, kwargs=kwargs))
        
         
    async def async_eval_module(self, module:str = None, fn:str='info', args:list = None, kwargs:dict=None, verbose:bool=False ):

        if args == None:
            args = []
        if kwargs == None:
            kwargs = {'timeout': self.config.timeout}

        if  kwargs == None:
            kwargs = {}
        if args == None:
            args = []

        if module == None:
            module = c.choice(self.module_names)
        module_state = self.name2module[module]
        w = 1
        emojis = c.emojis
        try:
            # get connection
            has_local_ip = any([k in module_state['address'].lower() for k in ['none', '0.0.0.0', '127.0.0.1', 'localhost']])
            if has_local_ip:
                raise Exception(f'Invalid address {module_state["address"]}')

            
            module_client = await c.async_connect(module_state['address'], network=self.config.network, namespace = self.namespace,timeout=1)
            response = await getattr(module_client,fn)(*args, **kwargs, return_future=True)
            w = self.score_response(response)

            c.print(f'{emojis["output"]} ITS LIT {response} {emojis["output"]} {emojis["dank"]} -> W : {w}', color='green',verbose=verbose)

        except Exception as e:
            response = {'error': str(e)}
            w = 0
            c.print(f'{module}::{fn} ERROR {emojis["error"]} {response} {emojis["error"]} -> W : {w}', color='red',verbose=verbose)
            


        self.count += 1
        module_stats = self.stats.get(module, module_state)
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*self.config.alpha + w*(1-self.config.alpha)

        module_stats['alpha'] = self.config.alpha
        module_stats['history'] = module_stats.get('history', []) + [{'input': dict(args=args, kwargs=kwargs) ,'output': response, 'w': w, 'time': c.time()}]
        self.stats[module] = module_stats

        if self.config.save_interval % self.count == 0:
            self.save()
        return module
    
    
    
    
    def save(self):
        tag = self.config.tag
        c.print(f'Saving stats to {tag}', color='white')
        tag = self.config.tag if tag == None else tag

        for k in self.stats:
            self.put(f'{tag}/stats/{k}', self.stats)



    def run(self, vote = True):
        

        self.running = True

        c.get_event_loop()
        
        while self.running:

            try:
                c.gather([self.async_eval_module() for i in range(self.config.parallel_jobs)], timeout=2)
            except Exception as e:
                c.print({'error': str(e)}, color='red')


            stats =  {
                'total_modules': self.count,
                'lifetime': int(self.lifetime),
                'modules_per_second': int(self.modules_per_second())

            }
            c.print(f'Validator Stats: {stats}', color='white')

           
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