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
            t = threading.Thread(target=self.run, kwargs={'vote':False, 'thread_id': t})
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


    @property
    def lifetime(self):
        return c.time() - self.start_time

    def modules_per_second(self):
        return self.count / self.lifetime


    def score_module(self, module) -> int:
        c.print('SCORE MODULE', color='green')
        return 1
        
    def eval_module(self, module = None, fn='info', args = None, kwargs=None, thread_id=0):
        return c.gather(self.async_eval_module(module=module, fn=fn, args=args, kwargs=kwargs, thread_id=thread_id,))
        
    async def async_eval_module(self, module:str = None, fn:str='info', args:list = None, kwargs:dict=None, verbose:bool=False , thread_id=0):

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
            # is it a local ip?, if it is raise an error
            has_local_ip = any([k in module_state['address'].lower() for k in ['none', '0.0.0.0', '127.0.0.1', 'localhost']])
            if has_local_ip:
                raise Exception(f'Invalid address {module_state["address"]}')
            

            # connect to module
            address  = module_state['address']
            module_client = await c.async_connect(address, network=self.config.network, namespace = self.namespace,timeout=1, key=self.key)

            # call function and return a future and await response
            response = await getattr(module_client,fn)(*args, **kwargs, return_future=True)

            # wait for response
            assert isinstance(response, dict), f'Response must be a dict, got {type(response)}'


            assert response.get('address', None) == module_state['address'] , f'Response must have an error key, got {response.keys()}'

            # get score from custom scoring function
            w = self.score_module(module=module_client)
            c.print(f'{emojis["output"]} ID:{thread_id} ITS LIT {response} {emojis["output"]} {emojis["dank"]} -> W : {w}', color='green',verbose=verbose)

        except Exception as e:
            # yall errored out, u get a gzero
            w = 0
            response = {'error': str(e)}
            c.print(f'{module}::{fn} ID:{thread_id} ERROR {emojis["error"]} {response} {emojis["error"]} -> W : {w}', color='red',verbose=verbose)
            


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

        if self.vote_staleness < self.config.voting_interval:
            return {'success': False, 'message': f'Voting too soon, wait {self.config.voting_interval - self.vote_staleness} seconds'}
        
        self.last_vote_time = c.time()
        self.load_stats()   
        topk = self.subnet['max_allowed_weights']

        vote_dict = {'uids': [], 'weights': []}

        for k, v in self.stats.items():
            vote_dict['uids'] += [v['uid']]
            vote_dict['weights'] += [v['w']]

        # get topk
        
        topk_indices = torch.argsort( torch.tensor(vote_dict['weights']), descending=True)[:topk].tolist()

        topk_indices = [i for i in topk_indices if vote_dict['weights'][i] > 0]
        if len(topk_indices) == 0:
            c.print('No modules to vote on', color='red')
            return {'success': False, 'message': 'No modules to vote on'}
        
        vote_dict['weights'] = [vote_dict['weights'][i] for i in topk_indices]
        vote_dict['uids'] = [vote_dict['uids'][i] for i in topk_indices]

        self.subspace.vote(uids=vote_dict['uids'],
                           weights=vote_dict['weights'], 
                           key=self.key, 
                           network=self.config.network, 
                           netuid=self.config.netuid)

        return {'success': True, 'message': 'Voted'}

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


    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_stats(self, k:str,default=None):
        if default == None:
            default = {}
        return self.get_json(f'stats/{self.config.network}/{k}', default=default)
    
    def save_module_stats(self,k:str, v):
        self.put_json(f'stats/{self.config.network}/{k}', v)

    @property
    def vote_staleness(self) -> float:
        return c.time() - self.last_vote_time


    @property
    def last_vote_time(self) -> float:
        return self.get('last_vote_time', 0)
    
    @last_vote_time.setter
    def last_vote_time(self, v:float):
        self.put('last_vote_time', v)


    def run(self, vote = True, thread_id = 0):

        if not self.subspace.is_registered(self.key):
            raise Exception(f'Key {self.key} is not registered in {self.config.network}')
            

        self.running = True
        c.get_event_loop()
        self.last_vote_time = c.time()
         
        import tqdm
        self.epochs = 0
        while self.running:

            modules = c.shuffle(c.copy(self.module_names))
            
            for i, module in enumerate(modules):

                if vote : 
                    self.vote()

                try:
                    self.eval_module(module=module, thread_id=thread_id)
                except Exception as e:
                    error = str(e)
                    if len(error) > 0:
                        c.print(f'Error in eval_module {e}', color='red')
                    self.errors += 1

                
                stats =  {
                    'total_modules': self.count,
                    'lifetime': int(self.lifetime),
                    'modules_per_second': int(self.modules_per_second()), 
                    'vote_staleness': self.vote_staleness,
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
    def serve(cls, key, remote=True,**kwargs):
        
        if remote:
            kwargs['remote'] = False
            return cls.remote_fn( fn='serve',  kwargs=kwargs)
        
        kwargs['start'] = False
        self = cls(**kwargs)
        self.start()

    @classmethod
    def start(cls, **kwargs):
        self = cls(**kwargs)
        self.run()

    def launch_worker(self,**kwargs):
        self.remote_fn( fn='start',   kwargs=kwargs)
