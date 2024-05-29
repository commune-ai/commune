
import commune as c
import os
from typing import *

class Vali(c.Module):

    voting_networks: ['subspace', 'bittensor']
    score_fns = ['score_module', 'score'] # the score functions
    whitelist = ['eval_module', 'score_module', 'eval', 'leaderboard']
    futures = []

    def __init__(self,
                 config:dict=None,
                 **kwargs):
        self.init_vali(config=config, **kwargs)

    def set_score_fn(self, score_fn: Union[Callable, str], module=None):
        """
        Set the score function for the validator
        """
        if score_fn != None:
            return {'success': True, 'msg': 'Default fn being used'}
        if isinstance(score_fn, str):
            fn = c.get_fn(score_fn)
        if not callable(score_fn):
            return {'success': False, 'msg': 'Score function must be callable'}
        self.score_module = score_fn
        return {'success': True, 'msg': 'Set score function', 'score_fn': self.score_fn.__name__}


    def init_vali(self, config=None, module=None, score_fn=None, **kwargs):
        # initialize the validator
        # merge the config with the default config
        config = self.set_config(config, kwargs=kwargs)
        config = c.dict2munch({**Vali.config(), **config})
        config.verbose = bool(config.verbose or config.debug)
        c.print(f'Initialized Vali with {config}', color='yellow')
        self.config = config
        self.init_metrics()
        self.set_score_fn(score_fn)

        c.thread(self.run_loop)

    init = init_vali

    def init_metrics(self):
        
        # COUNT METRICS
        self.requests = 0 
        self.errors = 0 
        self.successes = 0
        self.epochs = 0
        self.staleness_count = 0
        # timestamp metrics
        self.last_sync_time = 0
        self.last_error = 0
        self.last_sent = 0 
        self.last_success = 0


    @property
    def sent_staleness(self):
        return c.time()  - self.last_sent

    def epoch_info(self):

        return {
            'requests': self.requests,
            'errors': self.errors,
            'successes': self.successes,
            'sent_staleness': self.sent_staleness,
            'success_staleness': self.success_staleness,
            'staleness_count': self.staleness_count,
            'epochs': self.epochs,
            'executor_status': self.executor.status()
            

        }

    def start_workers(self):
        for i in range(self.config.workers):
            self.start_worker(i)


    @property
    def lifetime(self):
        return c.time() - self.start_time


    def run_loop(self):
        self.sync()
        c.sleep(self.config.initial_sleep)
        self.start_time = c.time()
        self.start_workers()
        too_stale_count = 0

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

                if self.success_staleness > self.config.max_success_staleness:
                    c.print('Too many stale successes, restarting workers', color='red')
                    self.start_workers()

                df = self.leaderboard()
                c.print(df.sort_values(by=['staleness'], ascending=False)[:42])
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



    def generate_finished_result(self):
        try:
            for future in c.as_completed(self.futures, timeout=self.config.timeout):
                result = future.result()
                self.futures.remove(future)  
                return result
        except Exception as e:
            result = c.detailed_error(e)
                
        return result


    def cancel_futures(self):
        for f in self.futures:
            f.cancel()

    epoch2results = {}

    def epoch(self,  **kwargs):

        module_addresses = c.shuffle(list(self.namespace.values()))
        c.print(f'Epoch {self.epochs} with {len(module_addresses)} modules', color='yellow')

        batch_size = min(self.config.batch_size, len(module_addresses)//4)
        self.executor = c.module('executor.thread')(max_workers=self.config.threads_per_worker,  maxsize=self.config.maxsize)
        progress_bar = c.tqdm(len(module_addresses))
        
        self.sync(network=self.config.network)

        results = []
        timeout = self.config.timeout

        for module_address in module_addresses:
            c.sleep(self.config.sample_interval)
            if not self.executor.is_full:
                future = self.executor.submit(self.eval, kwargs={'module': module_address},timeout=timeout)
                self.futures.append(future)
            if len(self.futures) >= batch_size:
                results.append(self.generate_finished_result())   

        self.cancel_futures()

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

    
    def sync_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     update = False,
                     fn : str = None,
                     **kwargs):

        if self.network_staleness() < self.config.sync_interval and not update:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'staleness': self.network_staleness(), 
                    'sync_interval': self.config.sync_interval,
                    'network': self.config.network, 
                    'netuid': self.config.netuid, 
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
            # local network does not need to be updated as it is atomically updated
            namespace = c.module('namespace').namespace(search=config.search, update=update)
        elif 'subspace' in config.network:
            if '.' in config.network:
                config.network, config.netuid = config.network.split('.')
            # convert the subnet to a netuid
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
        self.config = config
        
        return self.network_info()
    
    sync = set_network = sync_network

    @property
    def verbose(self):
        return self.config.verbose or self.config.debug

    def score_module(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        info = module.info()
        assert isinstance(info, dict), f'Info must be a dictionary, got {info}'
        return {'w': 1}
    
    def next_module(self):
        return c.choice(list(self.namespace.keys()))

    module2last_update = {}


    def get_module_path(self, module):
        if module in self.address2name:
            module = self.address2name[module]
        path = self.resolve_path(self.storage_path() +'/'+ module)
        return path


    def get_module(self, module:str, network:str='local', path=None, update=False, **kwargs):
        network = network or self.config.network
        self.sync(network=network, update=update, **kwargs)
        info = {}
        address = None
        # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
        if module in self.name2address:
            name = module
            address = self.name2address[module]
        else:
            assert module in self.address2name, f"{module} is not found in {self.config.network}"
            name = self.address2name[module]
            address = module
        path = self.get_module_path(module)
        module = c.connect(address, key=self.key)

        # CONNECT TO THE MODULE
        info = self.get(path, {})
        if 'ss58_address' not in info:
            info = module.info(timeout=self.config.timeout_info)
        
        info['past_timestamp'] = info.get('timestamp', 0) # for the stalnesss
        info['timestamp'] = c.timestamp() # the timestamp
        info['staleness'] = info['timestamp'] - info['past_timestamp']   
        info['w'] = info.get('w', 0) # the weight from the module
 
        if info['staleness'] < self.config.max_staleness:
            self.staleness_count += 1
            timeleft = self.config.max_staleness - info['staleness']
            raise {'module': info['name'], 'msg': 'Module is too new and w', 'staleness': info['staleness'], 'w': info['w'], 'timeleft': timeleft}

        info['past_w'] = info['w'] # for the alpha 
        info['path'] = path # path of saving the module
        info['name'] = name # name of the module cleint
        info['address'] = address # address of the module client
        info['alpha'] = self.config.alpha # ensure alpha is [0,1]
        setattr(module,'local_info', info) # set the client
        return module

    def eval(self, module:str, network:str=None, update=False, **kwargs):
        """
        The following evaluates a module sver
        """
        try:
            module = self.get_module(module=module, network=network, update=update)
            info = module.local_info
            self.last_sent = c.time()
            self.requests += 1
            response = self.score_module(module, **kwargs)
            response = self.process_response(response=response, info=info)
        except Exception as e:
            response = c.detailed_error(e)
            response['w'] = 0

            name = info.get('name', module)
            response_str = '('+' '.join([f"{k}={response[k]}" for k in ['line_text', 'line_no', 'file_name' ]]) + ')'
            c.print(f'Error (name={name}) --> {response_str}', color='red',  verbose=self.config.verbose)
            self.errors += 1
            self.last_error  = c.time()

        return response


    def process_response(self, response:dict, info:dict ):


        """

        Process the response from the score_module
        params:
            response
        """
        # PROCESS THE RESPONSE
        if type(response) in [int, float, bool]:
            # if the response is a number, we want to convert it to a dict
            response = {'w': float(response)}
        elif type(response) == dict:
            response = response
        else:
            raise Exception(f'Response must be a number or a boolean, got {response}')
        
        if not type(response['w']) in [int, float]:
            raise f'Response weight must be a number, got {response["w"]} with result : {response}'
        
        # merge response into modules info
        info.update(response)

        # resolve the alph
        info['latency'] = c.time() - info['timestamp']
        info['w'] = info['w']  * info['alpha'] + info['past_w'] * (1 - info['alpha'])
        
        # store modules that have a minimum weight to save storage of stale modules
        if info['w'] > self.config.min_leaderboard_weight:
            self.put(info['path'], info)

        c.print(f'Reward(w={info["w"]}, module={info["name"]} address={info["address"]} latency={c.round(info["latency"], 3)} staleness={info["staleness"]} )' , color='green')
        self.successes += 1
        self.last_success = c.time()

        return response


    @property
    def success_staleness(self):
        return c.time() - self.last_success

    
    eval_module = eval
        
    def storage_path(self, network=None):
        # the set storage path in config.path is set, then the modules are saved in that directory
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
    
    def calculate_votes(self, df=None):
        network = self.config.network
        keys = ['name', 'w', 'staleness','latency', 'ss58_address']
        leaderboard = df or self.leaderboard(network=network, 
                                       keys=keys, 
                                       to_dict=True)
        c.print(leaderboard)
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

    def vote(self,**kwargs):
        votes =self.calculate_votes() 
        weights = votes['weights']
        uids = votes['uids']
        return self.subspace.set_weights(uids=uids, # passing names as uids, to avoid slot conflicts
                            weights=weights, 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid,
                            **kwargs
                            )
    
    set_weights = vote 

    def module_info(self, **kwargs):
        if hasattr(self, 'subspace'):
            return self.subspace.module_info(self.key.ss58_address, netuid=self.config.netuid, **kwargs)
        else:
            return {}
    
    def leaderboard(self,
                    keys = ['name', 'w', 
                            'staleness',
                            'latency'],
                    max_age = None,
                    network = None,
                    ascending = True,
                    sort_by = ['w'],
                    to_dict = False,
                    n = None,
                    page = None,
                    **kwargs
                    ):
        max_age = max_age or self.config.max_leaderboard_age
        paths = self.module_paths(network=network)
        df = []
        # chunk the jobs into batches
        for path in paths:
            r = self.get(path, {},  max_age=max_age)
            if isinstance(r, dict) and 'ss58_address' and  r.get('w', 0) > self.config.min_leaderboard_weight  :
                r['staleness'] = c.time() - r.get('timestamp', 0)
                if not self.filter_module(r['name']):
                    continue
                df += [{k: r.get(k, None) for k in keys}]
            else :
                # removing the path as it is not a valid module and is too old
                self.rm(path)
        df = c.df(df) 
        if len(df) == 0:
            return c.df(df)
            
        df = df.sort_values(by=sort_by, ascending=ascending)

        if n != None:
            if page != None:
                df = df[page*n:(page+1)*n]
            else:
                df = df[:n]

        # if to_dict is true, we return the dataframe as a list of dictionaries
        if to_dict:
            return df.to_dict(orient='records')

        return df


    
    df = l = leaderboard
    
    def module_paths(self, network=None):
        paths = self.ls(self.storage_path(network=network))
        return paths
    
    def refresh_leaderboard(self):
        storage_path = self.storage_path()
        r = self.rm(storage_path)
        df = self.leaderboard()
        assert len(df) == 0, f'Leaderboard not removed {df}'
        return {'success': True, 'msg': 'Leaderboard removed', 'path': storage_path}
    
    def save_module_info(self, k:str, v:dict,):
        path = self.storage_path() + f'/{k}'
        self.put(path, v)
    

    def __del__(self):
        workers = self.workers()
        futures = []
        if self.config.mode == 'thread': 
            return [c.thread_map[w].cancel() for w in workers]
        elif self.config.mode == 'server':
            futures = [c.submit(c.kill, args=[w])  for w in workers]
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
