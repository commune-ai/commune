# import nest_asyncio
# nest_asyncio.apply()
import commune as c
import torch
import traceback
import threading 


class Validator(c.Module):

    def __init__(self, config=None,  **kwargs):
        self.set_config(config=config, kwargs=kwargs)
        c.print('BROOO')
        self.set_subspace( )
        self.module_stats = {}
        self.start_time = c.time()
        self.count = 0
        self.errors = 0
        
        if self.config.start:
            self.start()


    def kill_workers(self):
        for w in self.workers:
            c.kill(w)

    def start(self):
        self.threads = []
        # start threads, ensure they are daemons, and dont vote
        for t in range(self.config.num_threads):
            t = threading.Thread(target=self.run, kwargs={'vote':False, 'thread_id': t, 'vote': bool(t==0)})
            t.daemon = True
            t.start()

        # # main thread
        # self.run(vote=True)

    def set_subspace(self):
        self.subspace = c.module(self.config.network)()
        self.modules = self.subspace.modules()
        self.namespace = {v['name']: v['address'] for v in self.modules }
        self.name2module_state = {v['name']: v for v in self.modules } 
        self.module_names = list(self.name2module_state.keys())
        self.subnet = self.subspace.subnet()
        self.seconds_per_epoch = self.subspace.seconds_per_epoch()
        self.key = c.get_key(self.config.key)

    @property
    def lifetime(self):
        return c.time() - self.start_time

    def modules_per_second(self):
        return self.count / self.lifetime



    def resolve_module_name(self, module=None):
        if module == None:
            module = c.choice(self.module_names)
        return module

    def score_module(self, module):
        info = module.info(timeout=1)
        c.print(info)
        assert isinstance(info, dict), f'Response must be a dict, got {type(info)}'
        assert 'address' in info, f'Response must have an error key, got {info.keys()}'
        w = 1
        response = {'success': True,
                     'w': w}
        return response

    async def async_eval_module(self, module:str = None, thread_id=0, refresh=False):
        w = 0 # default weight is 0
        module_name = self.resolve_module_name(module)
        module_state = self.name2module_state[module_name]

        try:
            module = await c.async_connect(module_name,timeout=1, key=self.key, network=self.config.network)
            response = self.score_module(module)
            w = response['w']
            c.print(f'{module_name} SUCCESS {c.emojis["dank"]} -> W : {w}', color='green')
        except Exception as e:
            response = {'error': c.detailed_error(e), 'w': 0}
            c.print(f'{module_name} ERROR {c.emojis["error"]} -> W : {w}', color='red')

        w = response['w']
        module_stats = self.load_module_stats(module_name, default=module_state) if not refresh else module_state
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*self.config.alpha + w*(1-self.config.alpha)
        module_stats['alpha'] = self.config.alpha
        module_stats['history'] = module_stats.get('history', []) + [{'output': response, 'w': w, 'time': c.time()}]
        self.module_stats[module] = module_stats
        self.save_module_stats(module, module_stats)
        self.count += 1

        return module_stats
    
    def eval_module(self, module:str = None, thread_id=0, refresh=False):
        return c.gather(self.async_eval_module(module=module, thread_id=thread_id, refresh=refresh), timeout=2)
    

    def vote(self):

        if self.vote_staleness < self.config.voting_interval:
            return {'success': False, 'message': f'Voting too soon, wait {self.config.voting_interval - self.vote_staleness} seconds'}
        
        self.last_vote_time = c.time()
        self.load_stats()   
        topk = self.subnet['max_allowed_weights']

        vote_dict = {'uids': [], 'weights': []}

        for k, v in self.module_stats.items():
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
                self.module_stats[s['name']] = s
        return {'success': True, 'message': 'Loaded stats'}


    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_stats(self, k:str,default=None):
        default = {} if default == None else default
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
        c.print(f'Running -> thread:{thread_id} network:{self.config.network} netuid: {self.config.netuid} key: {self.key.path}', color='cyan')

        c.new_event_loop()

        self.running = True
        self.last_vote_time = c.time()
         
        import tqdm
        self.epochs = 0
        while self.running:
            modules = [m for m in self.module_names if m.startswith(self.config.module_prefix)]
            modules = c.shuffle(c.copy(modules))
            
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


    # def __del__(self):
    #     self.stop()
    #     self.save()
    #     c.print('Validator stopped', color='white')

        

    # def start_worker(self, **kwargs):
    #     config = self.config
    #     config.is_main_worker = False
    #     self = Validator(config=config)
    #     self.start()

    # def start_workers(self,**kwargs):

    #     for i in range(self.config.num_workers):
    #         name = self.name() + f'.w{i}'
    #         self.workers += [name]
    #         self.remote_fn( fn='start_worker', name=name,   kwargs=kwargs)
        # c.print('fam')
