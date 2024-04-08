
import commune as c
from typing import *

class Vali(c.Module):

    last_sync_time = 0
    last_sent = 0
    last_success = 0
    errors = 0
    requests = 0
    successes = 0  
    whitelist = ['eval_module', 'score_module']

    def __init__(self,
                 config:dict=None,
                 **kwargs):
        self.init(config=config, **kwargs)

    def init(self, config=None, module=None, **kwargs):
        if module != None:
            assert hasattr(module, 'score_module'), f'Module must have a config attribute, got {module}'
            self.score_module = module.score_module
        # initialize the validator
        config = self.set_config(config=config, kwargs=kwargs)
        # merge the config with the default config
        config = c.dict2munch({**Vali.config(), **config})
        c.print(config, 'VALI CONFIG')

        if hasattr(config, 'key'):
            self.key = c.key(config.key)
        self.config = config
        self.sync()
        c.thread(self.run_loop)

    init_vali = init


    def run_loop(self):
        c.sleep(self.config.initial_sleep)

        # start the workers
        
        self.start_time = c.time()
        for i in range(self.config.workers):
            self.start_worker(i)
        while True:

            c.sleep(self.config.sleep_interval)
            try:
                self.sync()
                run_info = self.run_info()
                c.print(run_info)

                r = {'success': False, 'msg': 'Vote Staleness is too low', 'vote_staleness': self.vote_staleness, 'vote_interval': self.config.vote_interval}
                if not 'subspace' in self.config.network and 'bittensor' not in self.config.network:
                    r = {'success': False, 'msg': 'Not a voting network', 'network': self.config.network}
                else:
                    if self.vote_staleness > self.config.vote_interval:
                        r = self.vote()
                run_info.update(r)
                c.print(run_info)

            except Exception as e:
                c.print(c.detailed_error(e))


    def epoch_info(self):
        info = {
            'requests': self.requests,
            'errors': self.errors,
            'successes': self.successes,
            'last_sent': c.round(c.time() - self.last_sent, 3),
            'last_success': c.round(c.time() - self.last_success, 3),
            'batch_size': self.config.batch_size,
            }
        return info

    def run_info(self):
        return {
            'network': self.network_info(),
            'epoch': self.epoch_info(),
            'vote': self.vote_info(),
            }
    
    def workers(self):
        if self.config.mode == 'server':
            return c.servers(search=self.server_name)
        elif self.config.mode == 'thread':
            return c.threads(search='worker')
        else:
            return []


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
    

    def epoch(self, batch_size = None, network=None, **kwargs):
        self.sync(network=network)
        futures = []
        results = []
        batch_size = batch_size or self.config.batch_size
        module_addresses = c.shuffle(list(self.namespace.values()))
        batch_size = min(batch_size, len(module_addresses)//2)
        self.executor = c.module('executor.thread')(max_workers=batch_size)
        batch_size = self.config.batch_size
        self.address2last_update = {}
        while len(module_addresses) > 0:
            module_address = module_addresses.pop()
            # if the futures are less than the batch, we can submit a new future
            futures.append(self.executor.submit(self.eval_module, args=[module_address], 
                                                timeout=self.config.timeout))

            if len(futures) >= batch_size:
                try:
                    for future in c.as_completed(futures,
                                                 timeout=self.config.timeout*2):
                        result = future.result()
                        c.print(result, verbose=self.config.debug or self.config.verbose)
                        futures.remove(future)
                        results += [result]  
                        break
                except Exception as e:
                    c.print(c.detailed_error(e))
                    
        if len(futures) >= 0:
            try:
                for future in c.as_completed(futures,
                                                timeout=self.config.timeout*2):
                    futures.remove(future)
                    result = future.result()
                    results += [result]  
                    c.print(result, verbose=self.config.debug)
            except Exception as e:
                c.print('ERROR',c.detailed_error(e))
        return results
        
    @property
    def sync_time(self):
        # return the time since the last sync with the network
        return c.time() - self.last_sync_time

    def is_voting_network(self):
        return 'subspace' in self.config.network or 'bittensor' in self.config.network
         

    def set_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     subnet: str = None,
                     fn : str = None,
                     max_age: int = 1000, **kwargs):
        
        if self.sync_time < self.config.sync_interval and (network == self.network and network != None):
            return {'msg': 'Alredy Synced network Within Interval', 
                    'sync_time': self.sync_time, 
                    'sync_interval': self.config.sync_interval,
                    'network': self.config.network, 
                    'subnet': self.config.netuid, 
                    'n': self.n,
                    'fn': self.config.fn,
                    'search': self.config.search,
                    }
        

        self.last_sync_time = c.time()
        # name2address / namespace
        network = network or self.config.network
        search =  search or self.config.search
        netuid =  netuid or self.config.netuid 
        fn = fn or self.config.fn        
        max_age = max_age or self.config.max_age

        
        if self.sync_time > self.config.sync_interval:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'last_sync_time': self.last_sync_time,
                    'sync_time': self.sync_time, 
                    'sync_interval': self.config.sync_interval,
                    }
        # RESOLVE THE VOTING NETWORKS
        if 'subspace' in network :
            # subspace network
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
            namespace = self.subspace.namespace(search=search, netuid=netuid, max_age=max_age)
        if 'bittensor' in network:
            # bittensor network
            self.subtensor = c.module('bittensor')(network=network, netuid=netuid)
            namespace = self.subtensor.namespace(search=search, netuid=netuid, max_age=max_age)
        if 'local' in network:
            # local network
            namespace = c.module('namespace').namespace(search=search, max_age=max_age)
    
        self.namespace = namespace
        self.n  = len(self.namespace)    
        self.name2address = self.namespace
        self.address2name = {v: k for k, v in self.namespace.items()}  

        self.network = network
        self.netuid = netuid
        self.fn = fn
        self.search = search
        self.max_age = max_age
        self.subnet = subnet

       
        return self.network_info()
    

    sync = set_network

    

    @property
    def verbose(self):
        return self.config.verbose or self.config.debug
    

    def process_response(self, response:dict):
        if type(response) in [int, float, bool]:
            # if the response is a number, we want to convert it to a dict
            response = {'w': float(response)}
        elif type(response) == dict:
            response = response
        else:
            raise Exception(f'Response must be a number or a boolean, got {response}')
        
        assert type(response['w']) in [int, float], f'Response weight must be a number, got {response["w"]}'
        return response
    

    def set_score_fn(self, score_fn):
        assert callable(score_fn), f'Score function must be callable, got {score_fn}'
        self.score_module = score_fn


    def score_module(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        ip = module.ls()
        assert isinstance(ip, dict), f"{module}, {ip}"
        return {'w': 1}
    

    def next_module(self):
        return self.random_module()
    
    def network_info(self):
        return {
                'search': self.config.search,
                'network': self.config.network, 
                'netuid': self.config.netuid, 
                'n': self.n,
                'fn': self.config.fn,
                'max_age': self.config.max_age,
                'sync_time': self.sync_time,
                }

    def eval_module(self, module:str = None, 
                    network=None, 
                    verbose = None,
                    verbose_keys = ['w', 'latency', 'name', 'address', 'ss58_address', 'path',  'staleness'],
                    catch_exception = True,
                    **kwargs):
        

        """
        The following evaluates a module sver
        """
        if catch_exception:
            try:
                kwargs = c.locals2kwargs(locals())
                kwargs['catch_exception'] = False
                return self.eval_module(**kwargs)
            except Exception as e:
            # give it 0
                self.errors += 1
                response =  c.detailed_error(e)
                response['w'] = 0
                return response

        verbose = verbose or self.verbose
        # load the module stats (if it exists)
        network = network or self.config.network
        self.sync(network=network)
        module = module or self.next_module()


        # load the module info and calculate the staleness of the module
        # if the module is stale, we can just return the module info
        self.requests += 1
        self.last_sent = c.time()

        info = {}

        # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
        if module in self.name2address:
            info['name'] = module
            info['address'] = self.name2address[module]
        else:
            assert module in self.address2name, f"{module} is not found in {self.network}"
            info['name'] = self.address2name[module]
            info['address'] = module
            
        # CONNECT TO THE MODULE
        module = c.connect(info['address'], key=self.key)

        path = self.resolve_path(self.storage_path() + f"/{info['name']}")
        cached_info = self.get(path, {}, max_age=self.config.max_age)

        if len(cached_info) > 0 :
            info = cached_info
        else:
            info = module.info(timeout=self.config.timeout)

        c.print(f'🚀 :: Eval Module {info["name"]} :: 🚀',  color='yellow', verbose=verbose)

        assert 'address' in info and 'name' in info, f'Info must have a address key, got {info.keys()}'
        info['staleness'] = c.time() - info.get('timestamp', 0)
        info['path'] = path

        start_time = c.time()
        response = self.score_module(module)
        response = self.process_response(response)
        response['timestamp'] = start_time
        response['latency'] = c.time() - response.get('timestamp', 0)
        # merge the info with the response, alpha is the smoothing factor
        response['w'] = response['w']  * self.config.alpha + info.get('w', response['w']) * (1 - self.config.alpha)
        # merge the info with the response
        info.update(response)
        self.put(path, info)
        response =  {k:info[k] for k in verbose_keys}

        self.successes += 1
        self.last_success = c.time()

        return response
        

    def storage_path(self, network=None):
        if self.config.get('path', None) != None:
            path = self.config.path
        else:
            network = network or self.config.network
            if 'subspace' in network:
                network_str = f'{network}.{self.netuid}'
            else:
                network_str = network
                
            path =  f'{network_str}'

        storage_path = self.resolve_path(path)

        return storage_path
        
    
    
    def resolve_tag(self, tag:str=None):
        return tag or self.config.vote_tag or self.tag
    
    def vote_info(self):
        votes = self.votes()
        if not self.is_voting_network():
            return {'success': False, 'msg': 'Not a voting network', 'network': self.config.network}

        info = {
            'num_uids': len(votes['uids']),
            'timestamp': votes['timestamp'],
            'staleness': self.vote_staleness,
            'key': self.key.ss58_address,
            'network': self.network,
        }
        return info
    
    
    def votes(self):
        network = self.config.network
        module_infos = self.module_infos(network=network, keys=['name', 'w', 'ss58_address'])
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        key2uid = self.subspace.key2uid() if hasattr(self, 'subspace') else {}
        for info in module_infos:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if 'ss58_address' in info and info['w'] >= 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid.get(info['ss58_address'], -1)]
        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes
    
    @property
    def votes_path(self):
        return self.storage_path() + f'/votes'

    def set_weights(self, 
                    uids:List[int]=None, 
                    weights: List[float]=None, **kwargs):
        if uids == None or weights == None:
            votes =self.votes() 
            weights = votes['weights']
            uids = votes['uids']

        if len(uids) < self.config.min_num_weights:
            return {'success': False, 
                    'msg': 'The votes are too low', 
                    'votes': len(votes['uids']), 
                    'min_num_weights': self.config.min_num_weights}
        
        return self.subspace.set_weights(uids=uids, # passing names as uids, to avoid slot conflicts
                            weights=weights, 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid,
                            **kwargs
                            )
    
    vote = set_weights
    



    def module_info(self, **kwargs):
        return self.subspace.get_module(self.key.ss58_address, netuid=self.netuid, **kwargs)
    
    def module_infos(self,
                    keys = ['name', 'w', 
                            'staleness',
                            'latency', 'ss58_address'],
                    path = 'cache/module_infos',
                    max_age = 3600,
                    network = None,
                    reverse = False,
                    sort_by = 'staleness',
                    df = False,
                    **kwargs
                    ):
        paths = self.module_paths(network=network)
        module_infos = []
        is_valid_module_info = lambda r : isinstance(r, dict) and 'ss58_address' in r
        # chunk the jobs into batches
        for path in paths:
            r = self.get(path, max_age=max_age)
            if is_valid_module_info(r):
                r['staleness'] = c.time() - r.get('timestamp', 0)
                module_infos += [{k: r.get(k, None) for k in keys}]
            else :
                self.rm(path)
        if sort_by != None and len(module_infos) > 0:
            module_infos = sorted(module_infos, key=lambda x: x[sort_by] if sort_by in x else 0, reverse=reverse)
        self.put(path, module_infos) 
        if df:
            module_infos = c.df(module_infos) 
            c.m('subspace')().serialize(module_infos)     
        return module_infos


    def num_modules(self, **kwargs):
        return len(self.leaderboard(**kwargs))

    def leaderboard(self, *args, df=True, **kwargs): 
        return self.module_infos(*args, df=df, **kwargs)
    
    def module_paths(self, network=None):
        paths = self.ls(self.storage_path(network=network))
        return paths
    
    def save_module_info(self, k:str, v:dict,):
        path = self.storage_path() + f'/{k}'
        self.put(path, v)
    

    def __del__(self):
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
    def test(cls, network='local', search='vali', n=4):
        modules = [c.serve(f'vali::{i}', network=network) for i in range(n)]
        c.serve('vali::test', kwargs=dict(network=network, search=search), wait_for_server=True)
        vali = c.connect('vali::test')


        while True:
            c.print(vali.run_info())
            c.sleep(3)

        return {'success': True, 'msg': 'Test Passed'}
            




    @property
    def vote_staleness(self):
        try:
            if 'subspace' in self.config.network:
                return self.subspace.block - self.module_info()['last_update']
        except Exception as e:
            pass
        return 0
    





        
Vali.run(__name__)
