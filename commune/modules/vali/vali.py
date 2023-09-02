# import nest_asyncio
# nest_asyncio.apply()
import commune as c
import torch
import traceback
import threading 
import queue




class Vali(c.Module):

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
        self.threads = []
        if self.config.start == False:
            return
        # we only need a queue if we are multithreading

        self.pool = c.module('process.pool')(fn=self.eval_module, num_workers=self.config.num_workers, verbose=self.config.verbose)

        # start threads, ensure they are daemons, and dont vote
        for t in range(self.config.num_workers):
            t = c.thread(fn=self.run_worker)
        # # main thread
        if self.config.vote:
            c.thread(self.vote_loop)
        

        c.thread(self.run)
            

    def kill_workers(self):
        for w in self.workers:
            c.kill(w)


    def sync(self):
        self.set_subspace(sync=True)

    def set_subspace(self, network=None, netuid=None, sync=False):
        if network == None:
            network = self.config.network
        if netuid == None:
            netuid = self.config.netuid
        if not hasattr(self, 'subsapce'):
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
        self.modules = self.subspace.modules(update=sync)
        self.namespace = {v['name']: v['address'] for v in self.modules }
        if self.config.module_prefix != None:
            self.modules = [m for m in self.modules if m['name'].startswith(self.config.module_prefix)]
        self.n  = len(self.modules)
        self.subnet = self.subspace.subnet()
        if self.config.vote_interval == None: 
            self.config['vote_interval'] = self.subspace.seconds_per_epoch()

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
        response = {'success': True,
                     'w': w}
        return response


    def eval_module(self, module:dict):
        epoch = self.count // self.n
        prefix = f'[bold cyan] [bold white]EPOCH {epoch}[/bold white] [bold yellow]SAMPLES :{self.count}/{self.n} [/bold yellow]'
        
        self.count += 1
        try:
            c.print(f'Calling {module["name"]}', color='cyan', verbose=self.config.verbose)
            module_client = c.connect(module['address'])
            response = self.score_module(module_client)
            c.print(f"Success {module['name']} {c.emojis['dank']} -> W : {response['w']}", color='green', verbose=self.config.verbose)
        except Exception as e:
            response = {'error': c.detailed_error(e), 'w': 0}
            c.print(f"BRUH  {module['name']} ERROR {c.emojis['error']} -> W : {response['w']} ->{response['error']['error']}", color='red', verbose=self.config.verbose)

        # the
        w = response['w']
        module_stats = self.load_module_stats( module['name'], default=module) if not refresh else module_state
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*(1-self.config.alpha) + w * self.config.alpha
        module_stats['timestamp'] = c.timestamp()
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
        path = cls.resolve_stats_path(network=network, tag=tag)
        return cls.rm(path)

    @classmethod
    def stats(cls, network='main', df:bool=True, keys=['name', 'w', 'count', 'staleness'], tag=None, topk=30):
        stats = cls.load_stats( network=network, keys=keys, tag=tag)

        if df:
            if len(stats) == 0:
                return c.df({'module': [], 'w': []})
            stats = c.df(stats)
            stats.sort_values('w', ascending=False, inplace=True)


        stats = stats[:topk]

        return stats


    @classmethod
    def weights(cls, network='main', df:bool=False, keys=['name', 'w', 'count', 'staleness', 'uid', 'key']):
        stats = cls.load_stats( network=network, keys=keys)
        weights = {s['name']: s['w'] for s in stats if s['w'] >= 0}


        if df:
            weights = c.df({'module': list(weights.keys()), 'w': list(weights.values())})
            weights.set_index('module', inplace=True)
            weights.sort_values('w', ascending=False, inplace=True)

        return weights

    def vote(self):
        stake = self.subspace.get_stake(self.key.ss58_address, netuid=self.config.netuid)

        if stake < self.config.min_stake:
            result = {'success': False, 'message': f'Not enough stake to vote, need at least {self.config.min_stake} stake'}
            return result
        elif self.vote_staleness < self.config.vote_interval:
            result = ({'success': False, 'message': f'Vote too soon, wait {self.config.vote_interval - self.vote_staleness} more seconds'})
            return result


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

        try:
            self.subspace.vote(uids=vote_dict['uids'],
                            weights=vote_dict['weights'], 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid)
        except Exception as e:
            response =  {'success': False, 'message': f'Error voting {e}'}
            c.print(response, color='red')
        self.last_vote_time = c.time()
        
        return {'success': True, 'message': 'Voted', 'votes': vote_dict }
    @classmethod
    def load_stats(cls, network:str='main', 
                    batch_size:int=20 , 
                    max_staleness:int= 2000,
                    keys:str=None,
                     tag=None):
        tag = 'base' if tag == None else tag
        paths = cls.ls(f'stats/{network}/{tag}')
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
        c.sleep(self.config.sleep_time)
        # we need a new event loop for each thread
        import asyncio
        loop = c.new_event_loop()

        while True:
            c.sleep(0.1)

            module = self.queue.get(block=True)
            start_time = c.time()
            # create a task for the event loop 
            # run the task
            loop.run_until_complete(asyncio.wait_for(self.async_eval_module(module=module), timeout=self.config.timeout))


    def vote_loop(self):
        c.sleep(self.config.sleep_time)
        while True:
            try:
                c.sleep(1)
                c.print(self.vote())
            except Exception as e:
                c.print(f'Error in vote loop {e}', color='red')
                c.print(traceback.format_exc(), color='red')
                c.sleep(1)


    def run(self, vote=False):
        c.sleep(self.config.sleep_time)
        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')

        c.new_event_loop()

        self.running = True

        while self.running:
            self.sync()
    
            modules = c.shuffle(c.copy(self.modules))
            for i, module in enumerate(modules):
                c.print(i)
                self.pool.submit(fn=self.eval_module, kwargs=dict(module=module))
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
        if hasattr(self, 'threads'):
            for t in self.threads:
                t.join(timeout=2)
        
    @classmethod
    def test(cls, **kwargs):
        kwargs['num_workers'] = 0
        kwargs['vote'] = False
        kwargs['verbose'] = True
        self = cls(**kwargs )
        return self.run()

