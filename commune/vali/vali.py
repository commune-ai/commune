
import commune as c

class Vali(c.Module):
    last_print = 0
    last_sync_time = 0
    last_sent = 0
    last_success = 0
    errors = 0
    count = 0
    requests = 0
    stale_requests = 0
    successes = 0
    epochs = 0
    n = 0    
    whitelist = ['eval_module', 'score_module']

    def __init__(self,config:dict=None,**kwargs):
        self.init_vali(config=config, **kwargs)

    def init_vali(self, config=None, **kwargs):
        # initialize the validator

        config = self.set_config(config=config, kwargs=kwargs)
        # merge the config with the default config
        self.config = c.dict2munch({**Vali.config(), **config})
        c.print(self.config)

        # start the workers
        self.sync()
        self.start_time = c.time()
        for i in range(self.config.workers):
            self.start_worker(i)
        c.thread(self.vote_loop)



    def run_info(self):
        info ={
            'vote_staleness': self.vote_staleness,
            'vote_interval': self.config.vote_interval,
            'successes': self.successes,
            'requests': self.requests,
            'stale_requests': self.stale_requests,
            'last_sent': c.round(c.time() - self.last_sent, 3),
            'last_success': c.round(c.time() - self.last_success, 3),
            'errors': self.errors,
            'block': self.subspace.block if 'subspace' in self.config.network else 0,
            'network': self.config.network,
            'subnet': self.config.netuid,
            }
        return info
    
    def workers(self):
        c.print(self.config, 'FAM')
        if self.config.mode == 'server':
            return c.servers(search=self.server_name)
        elif self.config.mode == 'thread':
            return c.threads(search='worker')
        else:
            return []
        
    def worker2logs(self):
        workers = self.workers()
        worker2logs = {}
        for w in workers:
            worker2logs[w] = c.logs(w, lines=100)

    @property
    def worker_name_prefix(self):
        return f'{self.server_name}'

    def restart_worker(self, id = 0):
        if self.config.mode == 'thread':
            return self.start_worker(id=id)

    def start_worker(self, id = 0, **kwargs):
        worker_name = self.worker_name(id)
        if self.config.mode == 'thread':
            worker = c.thread(self.worker, kwargs=kwargs, name=worker_name)
        elif self.config.mode == 'process':
            worker = c.process(self.worker, kwargs=kwargs, name=worker_name)
        elif self.config.mode == 'server':
            kwargs['config'] = self.config
            worker = self.serve(kwargs=kwargs, 
                                 key=self.key.path, 
                                 server_name = self.server_name + f'::{id}',)
        else:
            raise Exception(f'Invalid mode {self.config.mode}')
        
        c.print(f'Started worker {worker}', color='cyan')

        return {'success': True, 'msg': f'Started worker {worker}', 'worker': worker}

    def worker_name(self, id = 0):
        return f'{self.config.worker_fn_name}::{id}'

    def age(self):
        return c.time() - self.start_time


    def worker(self, 
               epochs=1e9,
               id=0):
        for epoch in range(int(epochs)): 
            try:
                self.epoch()
            except Exception as e:
                c.print('Dawg, theres an error in the epoch')
                c.print(c.detailed_error(e))

    @classmethod
    def run_epoch(cls, network='local', **kwargs):
        self = cls(workers=0, network=network, **kwargs)
        return self.epoch()
    

    def epoch(self, batch_size = None):

        self.executor = c.module('executor.thread')(max_workers=self.config.threads_per_worker)
        batch_size = batch_size or self.config.batch_size

        module_addresses = c.shuffle(list(self.namespace.values()))
        self.futures = []
        self.results = []
        timeout = self.config.timeout
        self.executor = c.module('executor.thread')(max_workers=batch_size)

        for module_address in module_addresses:
            # if the futures are less than the batch, we can submit a new future
            if len(self.futures) < batch_size:
                self.futures.append(self.executor.submit(self.eval_module, args=[module_address], timeout=timeout))
            else:
                try:
                    for future in c.as_completed(self.futures,timeout=timeout):
                        self.process_future(future)
                        break
                except Exception as e:
                    c.print(c.detailed_error(e))

        # wait for the remaining futures to complete
        if len(self.futures) > 0:
            try:
                for future in c.as_completed(self.futures,timeout=timeout):
                    self.process_future(future)
            except Exception as e:
                c.print(c.detailed_error(e))

        return self.results
        


    def clone_stats(self):
        workers = self.workers()
        stats = {}
        for w in workers:
            stats[w] = self.get(f'clone_stats/{w}', default={})
        return stats
    
    @property
    def time_since_sync(self):
        return c.time() - self.last_sync_time

    def is_voting_network(self):
        return 'subspace' in self.config.network or 'bittensor' in self.config.network
         

    def sync(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     subnet: str = None,
                     fn : str = None,
                     max_age: int = 1000, **kwargs):
        if self.config.sync_interval > self.time_since_sync:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'last_sync_time': self.last_sync_time,
                    'time_since_sync': self.time_since_sync, 
                    'sync_interval': self.config.sync_interval,
                    }
        # name2address / namespace
        network = self.network = self.config.network = network or self.config.network
        search = self.search =  self.config.search = search or self.config.search
        subnet = self.subnet = self.config.netuid =  self.netuid = netuid = netuid or subnet or self.config.netuid
        fn = self.fn = self.config.fn  = fn or self.config.fn        
        max_age = self.max_age = self.config.max_age = max_age or self.config.max_age

        # RESOLVE THE VOTING NETWORKS
        if 'subspace' in self.network :
            # subspace network
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
        if 'bittensor' in self.network:
            # bittensor network
            self.subtensor = c.module('bittensor')(network=network, netuid=netuid)


        self.namespace = c.module('namespace').namespace(search=search, 
                                    network=network, 
                                    netuid=netuid, 
                                    max_age=max_age)
    
        self.n  = len(self.namespace)    
        self.name2address = self.namespace
        self.address2name = {v: k for k, v in self.namespace.items()}    
        self.last_sync_time = c.time()

        r = {
                'search': search,
                'network': network, 
                'netuid': netuid, 
                'n': self.n, 
                }
        c.print(r)
        return r
    
    def score_module(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'

        return {'w': 1}

        
    def get_module_info(self, module):
        # if the module is in the namespace, we can just return the module info
        if module in self.name2address:
            module_name = module
            module_address = self.name2address[module]
        else:
            module_name = self.address2name.get(module, module)
            module_address = module

        info = self.load_module_info( module_name, {})
        if len(info) == 0:
            info = c.connect(module_address).info(timeout=self.config.timeout)
            info['timestamp'] = 0

        assert 'address' in info and 'name' in info
        return info
    
    def process_future(self, future):
        """
        The following processes the response
        """
        # check the response
        if future in self.futures:
            self.futures.remove(future)
        result = future.result()
        if result.get('w', 0) == 0:
            self.errors += 1
        else:
            self.successes += 1
            self.last_success = c.time()
        self.results += [result]
        return result

    
    def process_response(self, response):
        if type(response) in [int, float]:
            # if the response is a number, we want to convert it to a dict
            response = {'w': response}
        elif type(response) == bool:
            # if the response is a boolean, we want to convert it to a dict
            response = {'w': int(response)}
        elif type(response) == dict:
            pass
        else:
            raise Exception(f'Response must be a number or a boolean, got {response}')
        
        assert type(response['w']) in [int, float], f'Response weight must be a number, got {response["w"]}'


        return response
    @property
    def verbose(self):
        return self.config.verbose or self.config.debug


    def eval_module(self, module:str):
        """
        The following evaluates a module sver
        """
        # load the module stats (if it exists)

        
        # load the module info and calculate the staleness of the module
        # if the module is stale, we can just return the module info
        self.last_sent = c.time()
        self.requests += 1
        start_time = c.time()
        info = self.get_module_info(module)
        seconds_since_called = c.time() - info.get('timestamp', 0)
        if seconds_since_called < self.config.max_age:
            self.stale_requests += 1
            return {'w': info.get('w', 0),
                    'module': info['name'],
                    'address': info['address'],
                    'timestamp': c.time(), 
                    'msg': f'Module is too new with grace period being {self.config.max_age} > {int(seconds_since_called)}'}


        try:
            module = c.connect(info['address'], key=self.key)
            response = self.score_module(module)
            response = self.process_response(response)
            c.print(response, verbose=self.verbose)
        except Exception as e:
            # give it 0
            response = { 'w': 0,
                        'msg': f'{c.emoji("cross")} {info["name"]} {c.emoji("cross")}', 
                        'error': c.detailed_error(e)} 
        
        info.update(response)
        if info['w'] == 0:
            return info
        
        info['timestamp'] = start_time
        info['latency'] = c.time() - start_time
        info['w'] = response['w']  * self.config.alpha + info.get('w', response['w']) * (1 - self.config.alpha)
        info.update(response)            
        path = f'{self.storage_path()}/{info["name"]}'
        self.put_json(path, info)

        return {'w': info['w'], 'module': info['name'], 'address': info['address'], 'latency': info['latency']}
        

    def storage_path(self, network=None):
        network = network or self.config.network
        if 'subspace' in network:
            network_str = f'{network}.{self.netuid}'
        else:
            network_str = network
            
        path =  f'{network_str}'

        return path
        
    
    
    def resolve_tag(self, tag:str=None):
        return tag or self.config.vote_tag or self.tag
    
    def vote_info(self, votes = None):
        votes = votes or self.votes()
        info = {
            'num_uids': len(votes['uids']),
            'avg_weight': c.mean(votes['weights']),
            'stdev_weight': c.stdev(votes['weights']),
            'timestamp': votes['timestamp'],
            'lag': c.time() - votes['timestamp'],
        }
        return info
    
    def votes(self):
        network = self.config.network
        module_infos = self.module_infos(network=network, keys=['name', 'w', 'ss58_address'])
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        key2uid = self.subspace.key2uid()
        for info in module_infos:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if 'ss58_address' in info and info['w'] >= 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid[info['ss58_address']]]

        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes
    
    @property
    def votes_path(self):
        return self.storage_path() + f'/votes'

    def load_votes(self) -> dict:
        return self.get(self.votes_path, default={'uids': [], 'weights': [], 'timestamp': 0, 'block': 0})

    def save_votes(self, votes:dict):
        assert isinstance(votes, dict), f'Weights must be a dict, got {type(votes)}'
        assert 'uids' in votes, f'Weights must have a uids key, got {votes.keys()}'
        assert 'weights' in votes, f'Weights must have a weights key, got {votes.keys()}'
        assert 'timestamp' in votes, f'Weights must have a timestamp key, got {votes.keys()}'
        return self.put(self.votes_path, votes)





    def vote(self, async_vote:bool=False, save:bool = True, catch_exception=True, **kwargs):
        if catch_exception:
            try:
                return self.vote(async_vote=async_vote, save=save, catch_exception=False, **kwargs)
            except Exception as e:
                return c.detailed_error(e)
    
        if async_vote:
            future = c.submit(self.vote, **kwargs)
            return future

        votes =self.votes() 

        if len(votes['uids']) < self.config.min_num_weights:
            return {'success': False, 'msg': 'The votes are too low', 'votes': len(votes['uids']), 'min_num_weights': self.config.min_num_weights}

        r = c.vote(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                    weights=votes['weights'], 
                    key=self.key, 
                    network=self.config.network, 
                    netuid=self.config.netuid
                    )
        
        if save:
            self.save_votes(votes)

        
        return {'success': True, 
                'message': 'Voted', 
                'num_uids': len(votes['uids']),
                'avg_weight': c.mean(votes['weights']),
                'stdev_weight': c.stdev(votes['weights']),
                'saved': save,
                'r': r}
    
    def num_module_infos(self, **kwargs):
        return len(self.module_infos(**kwargs))

    @classmethod
    def leaderboard(cls, *args, **kwargs): 
        df =  c.df(cls.module_infos(*args, **kwargs))
        df.sort_values(by=['w', 'staleness'], ascending=False, inplace=True)
        return df
    
    def module_paths(self, network=None):
        paths = self.ls(self.storage_path(network=network))
        paths = list(filter(lambda x: x.endswith('.json'), paths))
        return paths
    

    @property
    def network_info(self):
        return {
            'network': self.config.network,
            'subnet': self.config.netuid,
            'fn': self.config.fn,
            'search': self.config.search,
            'max_age': self.config.max_age,
            'time_since_sync': c.time() - self.last_sync_time,
            'n': self.n,
        }
    

    @property
    def module_info(self):
        return self.subspace.get_module(self.key.ss58_address, netuid=self.netuid)
    
    def module_infos(self,
                    batch_size:int=100 , # batch size for 
                    timeout:int=10,
                    keys = ['name', 'w', 
                            'staleness', 'timestamp', 
                            'latency', 'address', 
                            'ss58_address'],
                    path = 'cache/module_infos',
                    max_age = 1000,
                    update = True,
                    network = None,
                    sort_by = 'staleness',
                    **kwargs
                    ):
        
        if not update:
            modules_info = self.get(path, default=[], max_age=max_age)
            if len(modules_info) > 0:
                return modules_info
            
        paths = self.module_paths(network=network)
        jobs = [c.async_get_json(p) for p in paths]
        module_infos = []
        # chunk the jobs into batches
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.wait(jobs_batch, timeout=timeout)
            for s in results:
                if isinstance(s, dict) and 'ss58_address' in s:
                    s['staleness'] = c.time() - s.get('timestamp', 0)
                    if s['staleness'] > max_age:
                        continue
                    module_infos += [{k: s.get(k, None) for k in keys}]

        if sort_by != None and len(module_infos) > 0:
            module_infos = sorted(module_infos, key=lambda x: x[sort_by] if sort_by in x else 0, reverse=True)
        if update:
            self.put(path, module_infos)       
        return module_infos

    def load_module_info(self, k:str,default=None):
        default = default if default != None else {}
        path = self.storage_path() + f'/{k}'
        return self.get_json(path, default=default)
    
    def save_module_info(self, k:str, v:dict):
        path = self.storage_path() + f'/{k}'
        self.put_json(path, v)

    def get_history(self, k:str, default=None):
        module_infos = self.load_module_info(k, default=default)
        return module_infos.get('history', [])
    
    
    
    def stop(self):
        self.running = False

    def __del__(self):
        self.stop()
        c.print(f'Vali {self.config.network} {self.config.netuid} stopped', color='cyan')
        workers = self.workers()
        futures = []
        for w in workers:
            if self.config.mode == 'thread': 
                c.print(f'Stopping worker {w}', color='cyan')
                futures += [c.submit(c.kill, args=[w])]
            elif self.config.mode == 'server':
                c.print(f'Stopping server {w}', color='cyan')
                futures += [c.submit(c.kill, args=[w])]
        return c.wait(futures, timeout=10)
        

    def random_module(self):
        return c.choice(list(self.namespace.keys()))

    @classmethod
    def test_eval_module(cls, network='local', verbose=False, timeout=1, workers=2, start=False,  **kwargs):
        self = cls(network=network, workers=workers, verbose=verbose, timeout=timeout, start=start,  **kwargs)
        return self.eval_module(self.random_module())


    @classmethod
    def test_network(cls, network='subspace', search='vali'):
        server_name = 'vali::test'
        self = cls(search=search, network=network, start=False, workers=0)
        if len(self.namespace) > 0:
            for module_name in self.namespace:
                assert search in module_name
        c.kill(server_name)
        return {'success': True, 'msg': f'Found {len(self.namespace)} modules in {network} {search}'}


    

    @property
    def vote_staleness(self):
        try:
            if 'subspace' in self.config.network:
                return self.subspace.block - self.module_info['last_update']
        except Exception as e:
            pass
        return 0
    
    
    def vote_loop(self):

        while True:
            c.print(self.run_info())
            c.sleep(self.config.sleep_interval)

            if self.vote_staleness < self.config.vote_interval:
                r = {'success': False, 'msg': 'Vote Staleness is too low', 'vote_staleness': self.vote_staleness, 'vote_interval': self.config.vote_interval}
            elif not 'subspace' in self.config.network and 'bittensor' not in self.config.network:
                r = {'success': False, 'msg': 'Not a voting network', 'network': self.config.network}
                c.print(r, color='red')
            else:
                c.print(f'Vali {self.config.network} {self.config.netuid} voting', color='cyan')
                r = self.vote(catch_exception=True)
                c.print(r)


        
Vali.run(__name__)
