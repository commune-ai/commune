# import nest_asyncio
# nest_asyncio.apply()
import commune as c
import torch
import traceback
import threading 
import queue




class Validator(c.Module):

    def __init__(self, config=None,  **kwargs):
        self.init_vali(config=config, **kwargs)
        if self.config.start:
            self.start()

    def init_vali(self, config=None, **kwargs):

        self.set_config(config=config, kwargs=kwargs)
        self.sync( )
        self.start_time = c.time()
        self.count = 0
        self.errors = 0
            
    def kill_workers(self):
        for w in self.workers:
            c.kill(w)

    def start(self):
        self.threads = []
        # we only need a queue if we are multithreading
        self.queue = c.queue(self.config.num_threads*4)
        # start threads, ensure they are daemons, and dont vote
        for t in range(self.config.num_threads):
            t = c.thread(fn=self.run_worker)
        # # main thread
        c.thread(self.vote_loop)
        c.thread(self.run)

    def sync(self, sync=True):
        self.subspace = c.module('subspace')(network=self.config.network, netuid=self.config.netuid)
        self.subspace.sync()
        self.modules = self.subspace.modules()
        self.namespace = {v['name']: v['address'] for v in self.modules }
        self.name2module = {v['name']: v for v in self.modules }
        self.module_names = [m for m in list(self.namespace.keys()) if m.startswith(self.config.module_prefix)]
        self.n  = len(self.module_names)
        self.subnet = self.subspace.subnet()
        self.seconds_per_epoch = self.subspace.seconds_per_epoch()
        self.key = c.get_key(self.config.key)

        if self.config.vote_interval == None: 
            self.config['vote_interval'] = self.seconds_per_epoch

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
        assert isinstance(info, dict), f'Response must be a dict, got {type(info)}'
        assert 'address' in info, f'Response must have an error key, got {info.keys()}'
        w = 1
        response = {'success': True,
                     'w': w}
        return response

    async def async_eval_module(self, module:str = None, thread_id=0, refresh=False):
        w = 0 # default weight is 0
        module_name = self.resolve_module_name(module)
        epoch = self.count // self.n
        prefix = f'[bold cyan]THREAD{thread_id}[bold cyan] [bold white]EPOCH {epoch}[/bold white] [bold yellow]SAMPLES :{self.count}/{self.n} [/bold yellow]'

        self.count += 1
        try:

            address = self.namespace.get(module_name)
            module = await c.async_connect(address,timeout=1)
            response = self.score_module(module)
            # c.print(f'{prefix} {module_name} SUCCESS {c.emojis["dank"]} -> W : {w}', color='green')
        except Exception as e:
            e = c.detailed_error(e)
            response = {'error': e, 'w': 0}
            # c.print(f'{prefix}  {module_name} ERROR {c.emojis["error"]} -> W : {w} ->..', color='red')

        w = response['w']
        module_info = self.name2module[module_name]
        module_stats = self.load_module_stats(module_name, default={}) if not refresh else module_state
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*(1-self.config.alpha) + w * self.config.alpha
        module_stats['name'] = module_name
        module_stats['key'] = module_info['key']
        module_stats['uid'] = module_info['uid']
        module_stats['timestamp'] = c.timestamp()
        module_stats['history'] = module_stats.get('history', []) + [{'output': response, 'w': w, 'time': c.time()}]
        self.save_module_stats(module_name, module_stats)

        return module_stats

    @classmethod
    def networks(cls):
        return [f.split('/')[-1] for f in cls.ls('stats')]

    @classmethod
    def resolve_stats_path(cls, network:str, tag:str=None):
        if tag == None:
            tag = 'base'
        return f'stats/{network}/{tag}'

    @classmethod
    def refresh_stats(cls, network='main', tag=None):
        path = cls.resolve_stats_path(network=network, tag=tag)
        return cls.rm(path)

    @classmethod
    def stats(cls, network='main', df=True, keys=['name', 'w', 'count', 'timestamp', 'uid', 'key']):
        self = cls(start=False)
        stats = cls.load_stats( network=network, keys=keys)

        if df:
            stats = c.df(stats)
            if len(stats) > 0:
                stats.sort_values('w', ascending=False, inplace=True)
        return stats

    def vote(self):
        if self.vote_staleness < self.config.vote_interval:
            return {'success': False, 'message': f'Vote too soon, wait {self.config.vote_interval - self.vote_staleness} more seconds'}

        self.last_vote_time = c.time()
        topk = self.subnet['max_allowed_weights']

        vote_dict = {'uids': [], 'weights': []}

        stats = self.stats(network=self.config.network, df=False, keys=['name', 'w', 'count', 'timestamp', 'uid', 'key'])
        vote_dict['uids'] = [v['uid'] for v in stats if v['w'] > 0] # get all uids where w > 0
        vote_dict['weights'] = [v['w'] for v in stats if v['w'] > 0] # get all weights where w > 0

        # get topk
        
        topk_indices = torch.argsort( torch.tensor(vote_dict['weights']), descending=True)[:topk].tolist()

        topk_indices = [i for i in topk_indices if vote_dict['weights'][i] > 0]
        if len(topk_indices) == 0:
            return {'success': False, 'message': 'No modules to vote on'}
        
        vote_dict['weights'] = [vote_dict['weights'][i] for i in topk_indices]
        vote_dict['uids'] = [vote_dict['uids'][i] for i in topk_indices]

        self.subspace.vote(uids=vote_dict['uids'],
                           weights=vote_dict['weights'], 
                           key=self.key, 
                           network=self.config.network, 
                           netuid=self.config.netuid)

        self.sync()
        
        return {'success': True, 'message': 'Voted'}
    @classmethod
    def load_stats(cls, network:str, batch_size=100, keys:str=True):
        paths = cls.ls(f'stats/{network}')
        jobs = [c.async_get_json(p) for p in paths]
        module_stats = []
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.gather(jobs_batch)
            for s in results:
                if s == None:
                    continue
                if keys :
                    s = {k: s.get(k,None) for k in keys}
                module_stats += [s]
        return module_stats


    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_stats(self, k:str,default=None):
        default = {} if default == None else default
        path = self.resolve_stats_path(network=self.config.network, tag=self.tag) + f'/{k}'
        return self.get_json(path, default=default)
    
    def save_module_stats(self,k:str, v):
        path = self.resolve_stats_path(network=self.config.network, tag=self.tag) + f'/{k}'
        self.put_json(path, v)

    @property
    def vote_staleness(self) -> int:
        return int(c.time() - self.last_vote_time)


    @property
    def last_vote_time(self) -> float:
        return self.get('last_vote_time', 0)
    
    @last_vote_time.setter
    def last_vote_time(self, v:float):
        self.put('last_vote_time', v)



    def run_worker(self):
        # we need a new event loop for each thread
        loop = c.new_event_loop()
        while True:
            try:
                c.sleep(0.1)
                module = self.queue.get()
                loop.run_until_complete(self.async_eval_module(module=module))
            except Exception as e:
                c.print(f'Error in worker {e}', color='red')
                traceback.print_exc()
                self.errors += 1


    def vote_loop(self):
        while True:
            c.sleep(1)
            if self.vote_staleness > self.config.vote_interval:
                self.vote()


    def run(self, vote=False):

        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')

        c.new_event_loop()

        self.running = True
        self.epochs = 0

        while self.running:
    
            

            modules = c.shuffle(c.copy(self.module_names))
            for i, module in enumerate(modules):
                if self.queue.full():
                    continue

                self.queue.put(module)
                if self.count % 100 == 0:
                    stats =  {
                    'total_modules': self.count,
                    'lifetime': int(self.lifetime),
                    'modules_per_second': int(self.modules_per_second()), 
                    'vote_staleness': self.vote_staleness,
                    'errors': self.errors,
                    'vote_interval': self.config.vote_interval,
                    'epochs': self.epochs,

                     }
                    c.print(f' --> {stats}\n', color='white')
                    # c.print(self.stats(network=self.config.network)[:10], color='white')
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
        self = cls(**kwargs, start=False)
        return self.eval_module()

