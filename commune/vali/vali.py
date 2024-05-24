
import commune as c
from typing import *

class Vali(c.Module):

    last_sync_time = 0
    voting_networks: ['subspace', 'bittensor']
    score_fns = ['score_module', 'score'] # the score functions
    whitelist = ['eval_module', 'score_module', 'eval', 'leaderboard']

    def __init__(self,
                 config:dict=None,
                 **kwargs):
        self.init_vali(config=config, **kwargs)

    def init_vali(self, config=None, module=None, score_fn=None, **kwargs):


        if score_fn != None:
            self.set_score_fn(score_fn)
        if module != None:
            for fn in self.score_fns:
                if hasattr(module, fn):
                    self.set_score_fn(getattr(module, fn))
        # initialize the validator
        # merge the config with the default config
        config = self.set_config(config, kwargs=kwargs)
        c.print(Vali.get_config())
        config = c.dict2munch({**Vali.get_config(), **config})
        if hasattr(config, 'key'):
            self.key = c.key(config.key)
        self.config = config
        c.print(f'Initialized Vali with {config}', color='yellow')
        self.sync()
        c.thread(self.run_loop)

    init = init_vali

    requests = 0 
    errors = 0 
    successes = 0
    last_sent = 0 
    last_success = 0
    epochs = 0

    @property
    def sent_staleness(self):
        return c.time()  - self.last_sent

    def epoch_info(self):

        return {
            'requests': self.requests,
            'errors': self.errors,
            'successes': self.successes,
            'sent_staleness': self.sent_staleness,
            'epochs': self.epochs,
            'executor_status': self.executor.status()

        }

    def start_workers(self):
        for i in range(self.config.workers):
            self.start_worker(i)


    def run_loop(self):
        c.sleep(self.config.initial_sleep)
        # start the workers
        self.start_time = c.time()
        self.start_workers()

        while True:
            c.sleep(self.config.print_interval)
            try:
                self.sync()
                run_info = self.run_info()
                if not 'subspace' in self.config.network and 'bittensor' not in self.config.network:
                    c.print({'success': False, 'msg': 'Not a voting network', 'network': self.config.network})
                else:
                    if self.vote_staleness > self.config.vote_interval:
                        c.print(self.vote())
                if run_info['epoch']['sent_staleness'] > self.config.max_sent_staleness:
                    self.start_workers()
                c.print(run_info)

            except Exception as e:
                c.print(c.detailed_error(e))

    def workers(self):
        if self.config.mode == None or str(self.config.mode) == 'server':
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
                                 key=self.server_name, 
                                 server_name = self.server_name + f'::{id}',)
        else:
            raise Exception(f'Invalid mode {self.config.mode}')
        
        c.print(f'Started worker {worker}', color='cyan')

        return {'success': True, 'msg': f'Started worker {worker}', 'worker': worker}

    def worker_name(self, id = 0):
        return f'worker::{id}'

    def age(self):
        return c.time() - self.start_time
    

 
    def worker(self, 
               epochs=1e9,
               id=0):
        for epoch in range(int(epochs)): 
            try:
                t0 = c.time()
                self.epoch()
                c.print(self.leaderboard())
                t1 = c.time()
                latency = t1 - t0
            except Exception as e:
                c.print('Dawg, theres an error in the epoch')
                c.print(c.detailed_error(e))

    @classmethod
    def run_epoch(cls, network='local', vali=None, **kwargs):
        if vali != None:
            cls = c.module('vali.'+vali)
        self = cls(network=network, **kwargs)
        return self.epoch()

    def epoch(self,  **kwargs):

        module_addresses = c.shuffle(list(self.namespace.values()))
        c.print(f'Epoch {self.epochs} with {len(module_addresses)} modules', color='yellow')
        batch_size = min(self.config.batch_size, len(module_addresses))
        self.executor = c.module('executor.thread')(max_workers=self.config.threads_per_worker, 
                                                    maxsize=self.config.threads_per_worker*2)
        progress_bar = c.tqdm(len(module_addresses))
        
        self.sync(network=self.config.network)

        futures = []
        results = []

        for module_address in module_addresses:
            c.sleep(self.config.sample_interval)
            future = self.executor.submit(self.eval, kwargs={'module': module_address},timeout=self.config.timeout)
            futures.append(future)

            if len(futures) >= batch_size:
                try:
                    for future in c.as_completed(futures, timeout=self.config.timeout):
                        result = future.result()
                        futures.remove(future)
                        results.append(result)
                        break
                except Exception as e:
                    pass
            
        for future in c.as_completed(futures, timeout=self.config.timeout):
            try:
                result = future.result()
            except Exception as e:
                result = c.detailed_error(e)
            futures.remove(future)
            results.append(result)

        return results
        
    def network_staleness(self):
        # return the time since the last sync with the network
        return c.time() - self.last_sync_time

    def is_voting_network(self):
        return 'subspace' in self.config.network or 'bittensor' in self.config.network
    
    def filter_module(self, module:str):
        if  self.config.search == None or self.config.search in module:
            return True
        return False
    
    def set_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     subnet: str = None,
                     fn : str = None,
                     **kwargs):
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
        self.last_sync_time = c.time()
        config = self.config
        # name2address / namespace
        config.network = network or config.network
        config.search =  search or config.search
        config.netuid =  netuid or config.netuid 
        # RESOLVE THE VOTING NETWORKS
        if 'local' in config.network:
            # local network
            namespace = c.module('namespace').namespace(search=config.search, max_age=config.sync_interval)
        elif 'subspace' in config.network:
            if '.' in config.network:
                config.network, config.netuid = config.network.split('.')
            if isinstance(config.netuid, str):
                config.netuid = self.subspace.subnet2netuid(config.netuid)
            self.subspace = c.module('subspace')(network=config.network)
            namespace = self.subspace.namespace(netuid=config.netuid, max_age=config.sync_interval)  
        else:
            raise Exception(f'Invalid network {config.network}')
        self.namespace = namespace
        self.namespace = {k: v for k, v in namespace.items() if self.filter_module(k)}

        self.n  = len(self.namespace)    
        self.name2address = self.namespace
        self.address2name = {k:v for v, k in namespace.items()}
        self.module2name = {v: k for k, v in self.namespace.items()}  

        self.network = network
        self.netuid = netuid
        self.fn = fn
        self.search = search
        self.subnet = subnet
        

        return self.network_info()
    

    sync = set_network

    

    @property
    def verbose(self):
        return self.config.verbose or self.config.debug


    def set_score_fn(self, score_fn):
        assert callable(score_fn), f'Score function must be callable, got {score_fn}'
        self.score_module = score_fn


    def score_module(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        info = module.info()
        assert isinstance(info, dict), f'Info must be a dictionary, got {info}'
        return {'w': 1}
    
    

    
    def next_module(self):
        return c.choice(list(self.namespace.keys()))

    module2last_update = {}

    def process_score(self, response):
        # PROCESS THE RESPONSE
        if type(response) in [int, float, bool]:
            # if the response is a number, we want to convert it to a dict
            response = {'w': float(response)}
        elif type(response) == dict:
            response = response
        else:
            raise Exception(f'Response must be a number or a boolean, got {response}')
        assert type(response['w']) in [int, float], f'Response weight must be a number, got {response["w"]}'

        return response

    def eval(self, module:str, 
            network:str=None, 
            verbose_keys:List[str] = ['w', 'latency', 'name', 'address', 'ss58_address', 'path',  'staleness'],
            **kwargs):
        """
        The following evaluates a module sver
        """
        # load the module stats (if it exists)
        network = network or self.config.network
        self.sync(network=network)

        info = {}
        # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
        if module in self.name2address:
            info['name'] = module
            info['address'] = self.name2address[module]
        else:
            assert module in self.address2name, f"{module} is not found in {self.config.network}"
            info['name'] = self.address2name[module]
            info['address'] = module

        # CONNECT TO THE MODULE
        module = c.connect(info['address'], key=self.key)
        path = self.resolve_path(self.storage_path() + f"/{info['name']}")
        info = self.get(path, {})
        is_info = bool(isinstance(info, dict) and 'ss58_address' in info)
        if not is_info:
            info = module.info(timeout=4)
        self.requests += 1
        name = info['name']
        address = info['address']
        key_address = info['ss58_address']
        c.print(f'Evaluating {name} (address={address},key={key_address})', color='purple')
        start_time = c.time()
        response = self.score_module(module)
        response = self.process_score(resuresponselt)
        response['timestamp'] = start_time
        response['latency'] = c.time() - response.get('timestamp', 0)
        response['w'] = response['w']  * self.config.alpha + info.get('w', response['w']) * (1 - self.config.alpha)
        response['w'] = c.round(response['w'], 3)
        # merge the info with the response
        if response['w'] > 0:
            self.successes += 1
            self.last_success = c.time()
            info.update(response)
            self.put(path, info)
        else:
            self.errors += 1
        response =  {k:info[k] for k in verbose_keys}
        c.print(f'Rewarding >>>> {result}',  color='green', verbose=self.config.verbose)
        return result
    
    eval_module = eval
        
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
        
    def vote_info(self):
        try:
            if not self.is_voting_network():
                return {'success': False, 'msg': 'Not a voting network', 'network': self.config.network}
            votes = self.calculate_votes()
        except Exception as e:
            votes = {'uids': [], 'weights': []}
            c.print(c.detailed_error(e))
        info = {
            'num_uids': len(votes.get('uids', [])),
            'staleness': self.vote_staleness,
            'key': self.key.ss58_address,
            'network': self.config.network,
        }
    
        return info
    
    def calculate_votes(self, 
                  
            ):
        network = self.config.network
        keys = ['name', 'w', 'staleness','latency', 'ss58_address']
        leaderboard = self.leaderboard(network=network, 
                                       keys=keys, 
                                       to_dict=True)
        assert len(leaderboard) > 0
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        key2uid = self.subspace.key2uid() if hasattr(self, 'subspace') else {}
        for info in leaderboard:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if 'ss58_address' in info and info['w'] >= 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid.get(info['ss58_address'], -1)]
        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes

    votes = calculate_votes
    
    @property
    def votes_path(self):
        return self.storage_path() + f'/votes'

    def set_weights(self,**kwargs):
        votes =self.calculate_votes() 
        weights = votes['weights']
        uids = votes['uids']
        if not self.config.disable_voting: 
            return {'success': False, 'msg': 'no voting mode enabled'}
        return self.subspace.set_weights(uids=uids, # passing names as uids, to avoid slot conflicts
                            weights=weights, 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid,
                            **kwargs
                            )
    
    vote = set_weights

    def module_info(self, **kwargs):
        if hasattr(self, 'subspace'):
            return self.subspace.module_info(self.key.ss58_address, netuid=self.config.netuid, **kwargs)
        else:
            return {}
    
    def leaderboard(self,
                    keys = ['name', 'w', 
                            'staleness',
                            'latency'],
                    path = 'cache/module_infos',
                    max_age = 3600,
                    min_weight = 0,
                    network = None,
                    ascending = True,
                    sort_by = ['staleness'],
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
                if not self.filter_module(r['name']):
                    continue
                df += [{k: r.get(k, None) for k in keys}]
            else :
                # removing the path as it is not a valid module and is too old
                self.rm(path)
        self.put(path, df) 
        df = c.df(df) 
        if len(df) == 0:
            return c.df(df)
        df = df.sort_values(by=sort_by, ascending=ascending)
        if min_weight > 0:
            df = df[df['w'] > min_weight]
        if n != None:
            if page != None:
                df = df[page*n:(page+1)*n]
            else:
                df = df[:n]

        # if to_dict is true, we return the dataframe as a list of dictionaries
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
    def test(cls, network='local', search='vali', n=4, sleep_time=5):
        # modules = [c.serve(f'vali::{i}', network=network) for i in range(n)]
        c.print(c.serve('vali::test', kwargs=dict(network=network, search=search), wait_for_server=True))
        leaderboard = c.call('vali::test/leaderboard')
        c.print(leaderboard)
        return {'success': True, 'msg': 'Test Passed'}
        

    @property
    def vote_staleness(self):
        try:
            if 'subspace' in self.config.network:
                return self.subspace.block - self.module_info()['last_update']
        except Exception as e:
            pass
        return 0
 

    def network_info(self):
        return {
            'search': self.config.search,
            'network': self.config.network, 
            'netuid': self.config.netuid, 
            'n': self.n,
            'fn': self.config.fn,
            'staleness': self.network_staleness(),

        }

    def run_info(self):
        return {
            'network': self.network_info(),
            'epoch': self.epoch_info() ,
            'vote': self.vote_info(),

            }
    
    @classmethod
    def check_peers(cls):
        servers = c.servers()
        module_path = cls.module_path()
        peers = [s for s in servers if s.startswith(module_path)]
        c.print(f'Found {len(peers)} peers')
        for peer in peers:
            c.print(f'Peer {peer} is alive')
            result = c.call(peer+'/run_info')
            c.print(result)


Vali.run(__name__)
