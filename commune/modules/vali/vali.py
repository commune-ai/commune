# import nest_asyncio
# nest_asyncio.apply()
import commune as c
import torch
import traceback
import threading 
import queue
import concurrent.futures
import gc
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed




class Vali(c.Module):

    whitelist = ['']
    last_sync_time = 0

    def __init__(self, config=None,  **kwargs):
        self.init_vali(config=config, **kwargs)



    def init_vali(self, config=None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        # merge the config with the default config
        self.count = 0
        self.config = c.munch({**Vali.config(), **config})
        self.start_time = c.time()
        self.errors = 0
        self.sync()
        self.process = c.module('process')
        self.ip = c.ip()
        if self.config.refresh_stats:
            self.refresh_stats(network=self.config.network, tag=self.tag)
        if self.config.start == False:
            return
        # # # # # main thread
        if self.config.vote:
            c.thread(self.vote_loop)
        c.thread(self.run)


        self.executor = c.module('thread.pool')(fn=self.eval_module, num_workers=self.config.num_workers)
            

    def kill_workers(self):
        for w in self.workers:
            c.kill(w)

    @property
    def sync_staleness(self):
        return int(c.time() - self.last_sync_time) 


    def sync(self, network:str=None, netuid:int=None, update: bool = True):
        
        try:
            if network == None:
                network = self.config.network
            if netuid == None:
                netuid = self.config.netuid
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
            self.modules = self.subspace.modules(update=False, netuid=netuid)
            self.namespace = {v['name']: v['address'] for v in self.modules }
            if self.config.module_prefix != None:
                self.modules = [m for m in self.modules if m['name'].startswith(self.config.module_prefix)]
            self.n  = len(self.modules)
            self.subnet = self.subspace.subnet()
            if self.config.vote_interval == None: 
                self.config['vote_interval'] = self.subspace.seconds_per_epoch()

            self.last_sync_time = c.time()
            self.block = self.subspace.block

            c.print('Syncing...', color='cyan')
        except Exception as e:
            c.print(f'Error syncing {e}', color='red')
            c.print(traceback.format_exc(), color='red')
            c.sleep(1)
            return



        return {'modules': self.modules, 'subnet': self.subnet}

    @property
    def lifetime(self):
        return c.time() - self.start_time

    def modules_per_second(self):
        return self.count / self.lifetime

    def score_module(self, module):

        info = module.info()
        assert isinstance(info, dict), f'Response must be a dict, got {type(info)}'
        assert 'address' in info, f'Response must have an error key, got {info.keys()}'
        w = 1
        response = {'success': True, 'w': w}
        return response


    def eval_module(self, module:dict):
        epoch = self.count // self.n
        prefix = f'[bold cyan] [bold white]EPOCH {epoch}[/bold white] [bold yellow]SAMPLES :{self.count}/{self.n} [/bold yellow]'
        
        self.count += 1

        my_module = self.ip in module['address']

        module_stats = self.load_module_stats( module['name'], default=module)

        staleness = c.time() - module_stats.get('timestamp', 0)
        if staleness < self.config.max_staleness:
            # c.print(f'{prefix} [bold yellow] {module["name"]} is too new as we pinged it {staleness}(s) ago[/bold yellow]', color='yellow')
            return


        try:
            module_client = c.connect(module['address'], key=self.key)
            response = self.score_module(module_client)
        except Exception as e:
            if my_module:
                c.print(f'{prefix} [bold red] {module["name"]} {e}[/bold red]', color='red')        
            response = {'error': c.detailed_error(e), 'w': 0}

        
        if my_module or response["w"] > 0:
            c.print(f'{prefix}[bold white]{c.emoji("dank")}{module["name"]}->{module["address"][:8]}.. W:{response["w"]}[/bold white] {c.emoji("dank")} ', color='green')
        
        w = response['w']
        response['timestamp'] = c.time()
        # we only want to save the module stats if the module was successful
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*(1-self.config.alpha) + w * self.config.alpha
        module_stats['timestamp'] = response['timestamp']
        # add the history of this module
        module_stats['history'] = module_stats.get('history', []) + [response]
        module_stats['history'] = module_stats['history'][-self.config.max_history:]

        self.save_module_stats(module['name'], module_stats)


        return module_stats

    @classmethod
    def networks(cls):
        return [f.split('/')[-1] for f in cls.ls('stats')]

    @classmethod
    def resolve_stats_path(cls, network:str, tag:str=None):
        if tag == None:
            tag = 'base'
        return f'stats/{network}/{tag}'
    def refresh_stats(self, network='main', tag=None):
        tag = self.tag if tag == None else tag
        path = self.resolve_stats_path(network=network, tag=tag)
        return self.rm(path)

    @classmethod
    def stats(cls, network='main', df:bool=True, keys=['name', 'w', 'count', 'staleness', 'address'], tag=None, topk=30):
        stats = cls.load_stats( network=network, keys=keys, tag=tag)

        if df:
            if len(stats) == 0:
                return c.df({'module': [], 'w': []})
            stats = c.df(stats)
            stats.sort_values(['w'], ascending=False, inplace=True)
            


        stats = stats[:topk]

        return stats


    @classmethod
    def votes(cls, network='main', tag=None):
        stats = cls.load_stats( network=network, keys=['uid', 'w'], tag=tag)
        votes = {
            'uids': [v['uid'] for v in stats if v['w'] > 0],  # get all uids where w > 0
            'weights': [v['w'] for v in stats if v['w'] > 0],  # get all weights where w > 0
            'timestamp': c.time()
        }
        return votes

    def vote(self):
        stake = self.subspace.get_stake(self.key.ss58_address, netuid=self.config.netuid)

        if stake < self.config.min_stake:
            result = {'success': False, 'message': f'Not enough stake to vote, need at least {self.config.min_stake} stake'}
            c.print(result, color='red')
            return result

        votes = self.votes(network=self.config.network, tag=self.tag)
        # get topk
        if len(votes['weights']) == 0:
            return {'success': False, 'message': 'No modules to vote on'}

        uids = c.copy(votes['uids'])
        new_votes = {'uids': [], 'weights': [], 'timestamp': c.time(), 'block': self.block}
        for i in range(len(votes['uids'])):
            if votes['uids'][i] < self.n :
                new_votes['uids'] += [votes['uids'][i]]
                new_votes['weights'] += [votes['weights'][i]]
        
        votes = new_votes
    

        topk = self.subnet['max_allowed_weights']
        topk_indices = torch.argsort( torch.tensor(votes['weights']), descending=True)[:topk].tolist()
        votes['weights'] = [votes['weights'][i] for i in topk_indices]
        votes['uids'] = [votes['uids'][i] for i in topk_indices]
        
        c.print(f'Voting on {len(votes["uids"])} modules', color='cyan')
        try:
            self.subspace.vote(uids=votes['uids'],
                            weights=votes['weights'], 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid)

            self.save_votes(votes)

        except Exception as e:
            response =  c.detailed_error(e)
            c.print(response, color='red')

        
        return {'success': True, 'message': 'Voted', 'votes': votes }

    @property
    def last_vote_time(self):
        votes = self.load_votes()
        return votes.get('timestamp', 0)

    def load_votes(self) -> dict:
        default={'uids': [], 'weights': [], 'timestamp': 0, 'block': 0}
        votes = self.get(f'votes/{self.config.network}.{self.tag}', default=default)
        return votes

    def save_votes(self, votes:dict):
        assert isinstance(votes, dict), f'Weights must be a dict, got {type(votes)}'
        assert 'uids' in votes, f'Weights must have a uids key, got {votes.keys()}'
        assert 'weights' in votes, f'Weights must have a weights key, got {votes.keys()}'
        assert 'timestamp' in votes, f'Weights must have a timestamp key, got {votes.keys()}'
        self.put(f'votes/{self.config.network}.{self.tag}', votes)

    

        

    @classmethod
    def saved_module_paths(cls, network:str='main', tag:str=None):
        tag = 'base' if tag == None else tag
        paths = cls.ls(f'stats/{network}/{tag}')
        return paths

    @classmethod
    def saved_module_names(cls, network:str='main', tag:str=None):
        paths = cls.saved_module_paths(network=network, tag=tag)
        modules = [p.split('/')[-1].replace('.json', '') for p in paths]
        return modules
        
    @classmethod
    def load_stats(cls, network:str='main', 
                    batch_size:int=20 , 
                    max_staleness:int= 2000,
                    keys:str=None,
                     tag=None):

        paths = cls.saved_module_paths(network=network, tag=tag)   
        jobs = [c.async_get_json(p) for p in paths]
        module_stats = []
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.gather(jobs_batch)


            # last_interaction = [r['history'][-1][] for r in results if r != None and len(r['history']) > 0]
            for s in results:
                if s == None:
                    continue
            

                if 'timestamp' in s:
                    s['staleness'] = c.timestamp() - s['timestamp']
                else:
                    s['staleness'] = 0
            
                if s['staleness'] > max_staleness:
                    continue

                if keys  != None:
                    s = {k: s.get(k,None) for k in keys}
                module_stats += [s]
        return module_stats


    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_stats(self, k:str,default=None):
        default = default if default != None else {}
        path = self.resolve_stats_path(network=self.config.network, tag=self.tag) + f'/{k}'
        return self.get_json(path, default=default)


    def get_history(self, k:str, default=None):
        module_stats = self.load_module_stats(k, default=default)
        return module_stats.get('history', [])
    
    def save_module_stats(self,k:str, v):
        path = self.resolve_stats_path(network=self.config.network) + f'/{k}'
        self.put_json(path, v)


    @property
    def vote_staleness(self) -> int:
        return int(c.time() - self.last_vote_time)


    def vote_loop(self):
        while True:
            if self.vote_staleness > self.config.vote_interval:
                self.vote()
            c.sleep(1)


    def run(self, vote=False):
        c.sleep(self.config.sleep_time)
        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')

        c.new_event_loop()

        self.running = True

        futures = []
        while self.running:

            if self.sync_staleness > self.config.sync_interval:
                self.sync()
            modules = c.shuffle(c.copy(self.modules))
            time_between_interval = c.time()
            for i, module in enumerate(modules):
                c.sleep(0.05)
                self.executor.submit(fn=self.eval_module, kwargs={'module':module})

                num_tasks = self.executor.num_tasks

                if self.count % 100 == 0 and self.count > 0:
                    stats =  {
                    'total_modules': self.count,
                    'lifetime': int(self.lifetime),
                    'modules_per_second': int(self.modules_per_second()), 
                    'vote_staleness': self.vote_staleness,
                    'errors': self.errors,
                    'vote_interval': self.config.vote_interval,
                    'epochs': self.epochs,
                     }
                    c.print(f'STATS  --> {stats}\n', color='white')
    @property
    def epochs(self):
        return self.count // self.n
    
           
    def check_score(self, module):
        module_name = module['name']

        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        
    @classmethod
    def test(cls, **kwargs):
        kwargs['num_workers'] = 0
        kwargs['vote'] = False
        kwargs['verbose'] = True
        self = cls(**kwargs )
        return self.run()

