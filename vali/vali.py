
import commune as c
from typing import *

class Vali(c.Module):
    last_sync_time = 0
    last_sent = 0
    last_success = 0
    errors = 0
    requests = 0
    successes = 0  
    score_fns = ['score', 'score']
    whitelist = ['eval_module', 'score', 'eval', 'leaderboard']
    address2last_update = {}

    def __init__(self,
                 config:dict=None,
                 **kwargs):
        self.init_vali(config=config, kwargs=kwargs)

    def init_vali(self, config=None, module=None, kwargs=None,  **extra_kwargs):
        if module != None:
            assert hasattr(module, 'score'), f'Module must have a config attribute'
            assert callable(module.score), f'Module must have a callable score attribute'
            self.score = module.score
        # initialize the validator
        # merge the config with the default config
        kwargs = kwargs or {}
        kwargs.update(extra_kwargs)
        config = self.set_config(config=config, kwargs=kwargs)
        config = c.dict2munch({**Vali.get_config(), **config})
        c.print(config, 'VALI CONFIG')
        if hasattr(config, 'key'):
            self.key = c.key(config.key)
        self.config = config
        # self.sync()
        c.thread(self.run_loop)

    init = init_vali



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
                run_info['vote'].update(r)

                df = self.leaderboard()[:10]
                c.print(df)
                c.print(run_info)

            except Exception as e:
                c.print(c.detailed_error(e))


    def network_info(self):
        return {
            'search': self.config.search,
            'network': self.config.network, 
            'netuid': self.config.netuid, 
            'n': self.n,
            'staleness': int(self.network_staleness()),

        }
    
    def epoch_info(self):
        return {
            'requests': self.requests,
            'errors': self.errors,
            'successes': self.successes,
            'last_sent': c.round(c.time() - self.last_sent, 3),
            'last_success': c.round(c.time() - self.last_success, 3),
            'batch_size': self.config.batch_size,
        }

    def run_info(self):
        return {
            'network': self.network_info(),
            'epoch': self.epoch_info() ,
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
        module_addresses = c.shuffle(list(self.namespace.values()))
        batch_size = min(self.config.batch_size, len(module_addresses))
        self.executor = c.module('executor.thread')(max_workers=batch_size)
        batch_size = self.config.batch_size
    
        for module_address in module_addresses:
            c.sleep(self.config.sample_sleep_interval)
            is_address = c.is_address(module_address)
            if not is_address:
                c.print(f'{module_address} is not a valid address', verbose=self.config.verbose)
                continue

            # if the futures are less than the batch, we can submit a new future
            lag = c.time() - self.address2last_update.get(module_address, 0)
            if lag < self.config.min_update_interval:
                # c.print(f'Module {module_address} is too fresh, skipping', verbose=self.config.debug)
                continue

            self.address2last_update[module_address] = c.time()
            futures.append(self.executor.submit(self.eval, args=[module_address],timeout=self.config.timeout))
            if len(futures) >= batch_size:
                try:
                    for future in c.as_completed(futures,
                                                 timeout=self.config.timeout):
                        result = future.result()
                        c.print(result, verbose=self.config.verbose)
                        futures.remove(future)
                        if c.is_error(result):
                            c.print('ERROR', result, verbose=self.config.verbose)
                            self.errors += 1
                        results += [result]  
                        break
                except Exception as e:
                    c.print('ERROR', c.detailed_error(e))

        if len(futures) >= 0:
            try:
                for future in c.as_completed(futures, timeout=self.config.timeout*2):
                    futures.remove(future) # remove the future
                    result = future.result() # result 
                    results += [result]  
                    c.print(result, verbose=self.config.debug)
            except Exception as e:
                c.print('ERROR',c.detailed_error(e))
        return results
        
    def network_staleness(self):
        # return the time since the last sync with the network
        return c.time() - self.last_sync_time

    def is_voting_network(self):
        return 'subspace' in self.config.network or 'bittensor' in self.config.network
    
    def filter_module(self, module:str):
        if self.config.search == None:
            return True
        if self.config.search in  module:
            return True
        return False
    
    def set_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     subnet: str = None,
                     fn : str = None,
                     max_age: int = 1000, **kwargs):
        
        if self.network_staleness() < self.config.sync_interval:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'staleness': self.network_staleness(), 
                    'sync_interval': self.config.sync_interval,
                    'network': self.config.network, 
                    'subnet': self.config.netuid, 
                    'n': self.n,
                    'fn': self.config.fn,
                    'search': self.config.search,
                    }

        config = self.config
        # name2address / namespace
        config.network = network or config.network
        config.search =  search or config.search
        config.netuid =  netuid or config.netuid 
        config.subnet = subnet or config.subnet or config.netuid
        config.fn = fn or config.fn        
        config.max_age_network = max_age or self.config.max_age_network

        self.last_sync_time = c.time()
        if self.network_staleness() > config.max_age_network:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'last_sync_time': self.last_sync_time,
                    'sync_lag': self.network_staleness(), 
                    'sync_interval': self.config.sync_interval,
                    }
        
        if '.' in  config.network:
            assert len(config.network.split('.')) == 2, f'Network must have one dot, got {config.network}'
            config.network, config.netuid = config.network.split('.')
            try:
                config.netuid = int(config.netuid)
            except:
                subnet2netuid = self.subspace.subnet2netuid()
                assert config.netuid in subnet2netuid, f'Netuid {config.netuid} not found in {subnet2netuid}'
                config.netuid = subnet2netuid[config.netuid]
                pass

        # RESOLVE THE VOTING NETWORKS
        if 'subspace' in config.network :
            self.subspace = c.module('subspace')(network=config.network, netuid=config.netuid)
            if isinstance(config.netuid, str):
                config.netuid = self.subspace.subnet2netuid(config.subnet)
            namespace = self.subspace.namespace(netuid=config.netuid, max_age=config.max_age_network)
        if 'bittensor' in config.network:
            self.subtensor = c.module('bittensor')(network=config.network, netuid=config.netuid)
            namespace = self.subtensor.namespace(netuid=config.netuid, max_age=config.max_age_network)
        if 'local' in config.network:
            # local network
            namespace = c.module('namespace').namespace(search=config.search, max_age=config.max_age_network)
    
        self.namespace = namespace
        self.namespace = {k: v for k, v in self.namespace.items() if self.filter_module(k)}
        c.print(f'Synced network {config.network} with {len(self.namespace)} modules', color='green')
        self.n  = len(self.namespace)    
        self.name2address = self.namespace
        self.address2name = {v: k for k, v in self.namespace.items()}  
        network_info = self.network_info()
        return network_info
    

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
        self.score = score_fn


    def score(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        info = module.info()
        assert isinstance(info, dict), f'Info must be a dictionary, got {info}'
        return {'w': 1}
    
    

    
    def next_module(self):
        return c.choice(list(self.namespace.keys()))
    

    def eval(self, module:str = None, 
                    network=None, 
                    verbose = None,
                    verbose_keys = None,
                    
                    **kwargs):
        


        """
        The following evaluates a module sver
        """
        verbose_keys = verbose_keys or ['w', 'latency', 'name', 'address', 'ss58_address', 'path',  'staleness']

        verbose = verbose or self.verbose
        # load the module stats (if it exists)
        network = network or self.config.network
        self.sync(network=network)
        module = module or self.next_module()
        c.print(f'🚀 :: Eval Module {module} :: 🚀',  color='yellow', verbose=verbose)


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

        info = self.get(path, {} , max_age=self.config.max_age_info)

        if len(info) == 0:
            info = module.info()
    

        assert 'address' in info and 'name' in info, f'Info must have a address key, got {info}'
        info['staleness'] = c.time() - info.get('timestamp', 0)
        info['path'] = path

        start_time = c.time()
        try:
            response = self.score(module)
            response = self.process_response(response)
        except Exception as e:
            response = c.detailed_error(e)
            response['w'] = 0
            verbose_keys = list(response.keys())

        response['timestamp'] = start_time
        response['latency'] = c.time() - response.get('timestamp', 0)
        response['w'] = response['w']  * self.config.alpha + info.get('w', response['w']) * (1 - self.config.alpha)
        for k in ['w', 'latency', 'timestamp']:
            info[k] = response[k]

        self.put(path, info)
        response =  {k:info[k] for k in verbose_keys}

        # record the success statistics
        if response['w'] > 0:
            self.successes += 1
        self.last_success = c.time()

        return response
    
    eval_module = eval
        

    def storage_path(self, network=None):
        if self.config.get('path', None) != None:
            path = self.config.path
        else:
            network = network or self.config.network
            if 'subspace' in network:
                network_str = f'{network}.{self.config.netuid}'
            else:
                network_str = network
                
            path =  f'{network_str}'

        storage_path = self.resolve_path(path)

        return storage_path
        
    
    
    def resolve_tag(self, tag:str=None):
        return tag or self.config.vote_tag or self.tag
    
    def vote_info(self):
        try:
            if not self.is_voting_network():
                return {'success': False, 'msg': 'Not a voting network', 'network': self.config.network}
            votes = self.votes()
        except Exception as e:
            votes = {'uids': [], 'weights': []}
            c.print(c.detailed_error(e))
        info = {
            'num_uids': len(votes.get('uids', [])),
            'vote_staleness': self.vote_staleness,
            'key': self.key.ss58_address,
            'network': self.config.network,
        }
    
        return info
    
    
    def votes(self, 
                  
            ):
        network = self.config.network
        keys = ['name', 'w', 'staleness','latency', 'ss58_address']
        leaderboard = self.leaderboard(network=network, keys=keys, to_dict=True)
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        is_voting_network = self.is_voting_network()
        key2uid = {}
        self_uid = None
        if self.config.network in ['subspace']:
            self_uid =  self.subspace.get_uid(key=self.key.ss58_address, 
                                                network=self.config.network, 
                                                netuid=self.config.netuid)
            key2uid = self.subspace.key2uid(netuid=self.config.netuid)
        for info in leaderboard:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if  isinstance(info, dict) and 'ss58_address' in info:
                if (info['ss58_address'] in key2uid and info['w'] >= 0) or not is_voting_network:
                    uid = key2uid.get(info['ss58_address'], -1)
                    if self_uid == None or uid != self_uid:
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
                    weights: List[float]=None, 
                    **kwargs):
        self.sync()
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
        return self.subspace.module_info(self.key.ss58_address, netuid=self.config.netuid, **kwargs)
    
    def leaderboard(self,
                    keys = ['name', 'w', 'staleness','latency'],
                    path = 'cache/module_infos',
                    max_age = 3600,
                    min_weight = 0,
                    network = None,
                    ascending = False,
                    sort_by = ['w'],
                    to_dict = False,
                    n = 50,
                    page = None,
                    **kwargs
                    ):
        if hasattr(self.config, 'max_leaderboard_age'):
            max_age = self.config.max_leaderboard_age
        paths = self.module_paths(network=network)
        df = []
        # chunk the jobs into batches
        for path in paths:
            r = self.get(path, max_age=max_age)
            if isinstance(r, dict) and 'ss58_address' in r:
                r['staleness'] = c.time() - r.get('timestamp', 0)
                df += [{k: r.get(k, None) for k in keys}]
            else :
                # removing the path as it is not a valid module and is too old
                self.rm(path)

        self.put(path, df) 
        df = c.df(df) 
        assert len(df) > 0
        # sort by w in the up and staleness in the down direction
        df = df.sort_values(by=sort_by, ascending=ascending)
        if min_weight > 0:
            df = df[df['w'] > min_weight]
        if n != None:
            if page != None:
                df = df[page*n:(page+1)*n]
            else:
                df = df[:n]
        if to_dict:
            return df.to_dict(orient='records')
        return df


    
    l = leaderboard
    
    def module_paths(self, network=None):
        paths = self.ls(self.storage_path(network=network))
        return paths
    
    def save_module_info(self, k:str, v:dict,):
        path = self.storage_path() + f'/{k}'
        self.put(path, v)
    

    def __del__(self):
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
