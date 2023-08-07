# import nest_asyncio
# nest_asyncio.apply()
import commune as c
import torch

import threading 


class Validator(c.Module):

    def __init__(self, config=None,  **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        self.set_subspace( )
        self.stats = {}
        self.start_time = c.time()
        self.count = 0
        self.errors = 0
        
        # c.print(c.key)
        
        if self.config.start:
            self.start()

    def start(self):
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
        self.subnet = self.subspace.subnet()
        self.seconds_per_epoch = self.subspace.seconds_per_epoch()
    
        self.key = c.get_key(self.config.key)
        if not self.subspace.is_registered(self.key):
            raise Exception(f'Key {self.key} is not registered in {self.config.network}')


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
            c.print(f'{module}::{fn} ERROR {emojis["error"]} {response} {emojis["error"]} -> W : {w}', color='red',verbose=verbose)
            w = 0
            c.print(f'{module}::{fn} ERROR {emojis["error"]} {response} {emojis["error"]} -> W : {w}', color='red',verbose=verbose)
            


        module_stats = self.load_module_stats(module, module_state)
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*self.config.alpha + w*(1-self.config.alpha)
        module_stats['alpha'] = self.config.alpha
        module_stats['history'] = module_stats.get('history', []) + [{'input': dict(args=args, kwargs=kwargs) ,'output': response, 'w': w, 'time': c.time()}]
        self.stats[module] = module_stats
        self.save_module_stats(module, module_stats)
        self.count += 1

        return module
    

    def vote(self):
        self.last_vote_time = c.time()
        topk = self.subnet['max_allowed_weights']

        self.load_stats()   
        vote_dict = {'uids': [], 'weights': []}

        for k, v in self.stats.items():
            vote_dict['uids'] += [v['uid']]
            vote_dict['weights'] += [v['w']]

        
        # get topk
        
        topk_indices = torch.argsort( torch.tensor(vote_dict['weights']), descending=True)[:topk].tolist()
        vote_dict['weights'] = [vote_dict['weights'][i] for i in topk_indices]
        vote_dict['uids'] = [vote_dict['uids'][i] for i in topk_indices]

        self.subspace.vote(uids=vote_dict['uids'],
                           weights=vote_dict['weights'], 
                           key=self.key, 
                           network=self.config.network, 
                           netuid=self.config.netuid)

        return vote_dict


    def load_stats(self, batch_size=100):
        paths = self.ls(f'stats/{self.config.network}')
        jobs = [c.async_get_json(p) for p in paths]

        for jobs_batch in c.chunk(jobs, batch_size):
            module_stats = c.gather(jobs_batch)
            for s in module_stats:
                if s == None:
                    continue
                self.stats[s['name']] = s
        return {'success': True, 'message': 'Loaded stats'}


    
    def load_module_stats(self, k:str,default=None):
        if default == None:
            default = {}
        return self.get_json(f'stats/{self.config.network}/{k}', default=default)
    
    def save_module_stats(self,k:str, v):
        self.put_json(f'stats/{self.config.network}/{k}', v)



    def run(self, vote = True):
        

        self.running = True

        c.get_event_loop()
        self.last_vote_time = c.time()
         
        import tqdm
        self.epochs = 0
        while self.running:

            modules = c.shuffle(self.module_names)
            
            for i, module in enumerate(modules):

                vote_staleness = c.time() - self.last_vote_time

                if vote_staleness > self.config.voting_interval:

                    self.vote()


                try:
                    self.eval_module(module=module)
                except Exception as e:
                    c.print(f'Error in eval_module {e}', color='red')
                    self.errors += 1

                
                stats =  {
                    'total_modules': self.count,
                    'lifetime': int(self.lifetime),
                    'modules_per_second': int(self.modules_per_second()), 
                    'vote_staleness': vote_staleness,
                    'errors': self.errors,

                }
                if self.count % 100 == 0:
                    c.print(f'Validator Stats: {stats}', color='white')
            self.epochs += 1
    
           
    def check_score(self, module):
        module_name = module['name']

        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        if hasattr(self, 'threads'):
            for t in self.threads:
                t.join(timeout=2)
        
    @classmethod
    def test(cls, **kwargs):
        self = cls(**kwargs)


    def __del__(self):
        self.stop()
        self.save()
        c.print('Validator stopped', color='white')

    @classmethod
    def serve(cls, key, remote=True, **kwargs):
        kwargs = c.locals2kwargs(locals())
        if remote:
            kwargs['remote'] = False
            return cls.remote_fn( fn='serve', name=f'vali::default::{key}', kwargs=kwargs)
        
        kwargs['start'] = False
        self = cls(**kwargs)
        self.start()