
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):

    whitelist = ['eval_module', 'score_module', 'eval', 'leaderboard']

    def __init__(self,
                 config:dict=None,
                 **kwargs):
        self.init_vali(config=config, **kwargs)

    def set_score_fn(self, score_fn: Union[Callable, str] = None, module=None):
        """
        Set the score function for the validator
        """
        module = module or self 
        score_fn_options = []
        if score_fn == None:
            for fn in self.config.score_fns:
                if hasattr(module, fn):
                    score_fn_options += [fn]
            score_fn =  score_fn_options[0]
            assert len(score_fn_options) == 1, f'Bruhhhh, you have multiple score functions {score_fn_options}, choose one fam'
        else:
            if isinstance(score_fn, str):
                score_fn = c.get_fn(score_fn)
        self.score = getattr(self, score_fn)
        return {'success': True, 'msg': 'Set score function', 'score_fn': self.score.__name__}

    def init_state(self):
        # COUNT METRICS
        self.last_start_time = 0 # the last time a worker was started
        self.requests = 0  # the number of requests
        self.errors = 0  # the number of errors
        self.successes = 0 # the number of successes
        self.epochs = 0 # the number of epochs
        self.staleness_count = 0 # the number of staleness
        # timestamp metrics
        self.last_sync_time = 0 # the last time the network was synced
        self.last_error = 0 # the last time an error occured
        self.last_sent = 0  # the last time a request was sent
        self.last_success = 0 # the last time a success was made
        self.start_time = c.time()
        self.results = []
        self.futures = []

    def set_module(self, 
                   module:str, 
                   **kwargs):

        if isinstance(module, str):
            module = c.module(module, **kwargs)()
        does_module_have_score_fn = any([hasattr(module, fn) for fn in self.config.score_fns])
        assert does_module_have_score_fn, f'Module must have a score function, got {module}'
        if hasattr(module, 'storage_dir'):
            storage_path = self.storage_dir()
        else:
            storage_path = module.__class__.__name__
        self.config.storage_path = storage_path
        self.module = module

    def init_vali(self, config=None, module=None, score_fn=None, **kwargs):
        # initialize the validator
        # merge the config with the default config
        if module != None:
            self.set_module(module, **kwargs)
        config = self.set_config(config, kwargs=kwargs)
        config = c.dict2munch({**Vali.config(), **config})
        config.verbose = bool(config.verbose or config.debug)
        self.config = config

        self.init_state()
        self.set_score_fn(score_fn)
        self.set_network(network=self.config.network, 
                     search=config.search,  
                     netuid=config.netuid, 
                     max_age = config.max_network_staleness)
        # start the run loop
        if self.config.run_loop:
            c.thread(self.run_loop)
        
    init = init_vali

    @property
    def sent_staleness(self):
        return c.time()  - self.last_sent

    @property
    def success_staleness(self):
        return c.time() - self.last_success

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
        c.print('Starting workers', color='cyan')
        for i in range(self.config.workers):
            self.start_worker(i)


    @property
    def lifetime(self):
        return c.time() - self.start_time
    
    @property
    def is_voting_network(self):
        return any([v in self.config.network for v in self.config.voting_networks])
    
    @property
    def last_start_staleness(self):
        return c.time() - self.last_start_time

    def run_step(self):
        """
        The following runs a step in the validation loop
        """

        self.sync()
        run_info = self.run_info()
        should_start_workers = self.success_staleness > self.config.max_success_staleness and self.last_start_staleness > self.config.max_success_staleness
        if should_start_workers:
            c.print('Too many stale successes, restarting workers', color='red')
            self.start_workers()
        if self.is_voting_network and self.vote_staleness > self.config.vote_interval:
            buffer = '='*10
            c.print(buffer+'Voting'+buffer, color='cyan')
            c.print(self.vote())
        c.print(run_info)
        c.print(buffer+f'Epoch {self.epochs} with {self.n} modules', color='yellow'+buffer)
        c.print(self.leaderboard())

    def run_loop(self):
        """
        The run loop is a backgroun loop that runs to do two checks
        - network: check the staleness of the network to resync it 
        - workers: check the staleness of the last success to restart the workers 
        - voting: check the staleness of the last vote to vote (if it is a voting network)
        
        """
        self.sync()
        # start the workers
        self.start_workers()

        while True:
            c.sleep(self.config.run_step_interval)
            try:
                self.run_step()
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
        self.last_start_time = c.time()
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
               epochs=1e9):
        for epoch in range(int(epochs)): 
            try:
                result = self.epoch()
            except Exception as e:
                result = c.detailed_error(e)
            c.print(f'Leaderboard epoch={self.epochs})'  , color='yellow')



    def generate_finished_result(self):
        try:
            for future in c.as_completed(self.futures, timeout=self.config.timeout):
                self.futures.remove(future) 
                result = future.result()
                emoji =  '🟢' if result['w'] > 0 else '🔴'

                if c.is_error(result):
                    msg = ' '.join([f'{k}={result[k]}' for k in result.keys()])
                    msg =  f'ERROR({msg})'
                else:
                    result = {k: result[k] for k in self.config.result_keys}
                    msg = ' '.join([f'{k}={result[k]}' for k in result])
                    msg = f'SUCCESS({msg})'
                break
        except Exception as e:
            emoji = '🔴'
            result = c.detailed_error(e)
            msg = f'Error({result})'
        c.print(emoji + msg + emoji, 
                color='cyan', 
                verbose=self.config.verbose)
        return result


    def cancel_futures(self):
        for f in self.futures:
            f.cancel()

    epoch2results = {}

    @classmethod
    def run_epoch(cls, network='local', vali=None, run_loop=False,  **kwargs):
        if vali != None:
            cls = c.module(vali)
        self = cls(network=network, run_loop=run_loop, **kwargs)
        return self.epoch()

    def epoch(self,  **kwargs):
    
        try:
            self.epochs += 1
            self.sync_network(**kwargs)
            module_addresses = c.shuffle(list(self.namespace.values()))
            c.print(f'Epoch {self.epochs} with {len(module_addresses)} modules', color='yellow')
            batch_size = min(self.config.batch_size, len(module_addresses)//4)
            self.executor = c.module('executor.thread')(max_workers=self.config.threads_per_worker,  maxsize=self.config.max_size)
            
            self.sync(network=self.config.network)

            results = []
            timeout = self.config.timeout
            n = len(module_addresses)
            self.current_epoch = self.epochs
            c.print(f'Starting epoch {self.current_epoch} with n={n}',)

            for module_address in module_addresses:
                if not self.executor.is_full:
                    future = self.executor.submit(self.eval, kwargs={'module': module_address},timeout=timeout)
                    self.futures.append(future)
                if len(self.futures) >= batch_size:
                    result = self.generate_finished_result()
                    results.append(result)


            while len(self.futures) > 0:
                result = self.generate_finished_result()
                results.append(result)

        except Exception as e:
            c.print(c.detailed_error(e), color='red')

        results = [r for r in results if not c.is_error(r)]
        print(len(results))
        if len(results) > 0 and 'w' in results[0]:
            df =  c.df(results)
            df = df.sort_values(by='w', ascending=False)
            return df
        else:
            return results


    @property
    def network_staleness(self) -> int:
        """
        The staleness of the network
        """
        return c.time() - self.last_sync_time

    def filter_module(self, module:str, search=None):
        search = search or self.config.search
        if ',' in str(search):
            search_list = search.split(',')
        else:
            search_list = [search]
        return all([s == None or s in module  for s in search_list ])

    
    def set_network(self, 
                     network:str=None, 
                     netuid:int=None,**kwargs):
        config = self.config
        config.network = network or config.network
        config.netuid =  netuid or config.netuid
        if len(kwargs) > 0:
            config = c.dict2munch({**config, **kwargs})
            
        if self.network_staleness < config.max_network_staleness:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'staleness': self.network_staleness, 
                    'max_network_staleness': self.config.max_network_staleness,
                    'network': self.config.network, 
                    'netuid': self.config.netuid, 
                    'n': self.n,
                    'search': self.config.search,
                    }
        c.print(f'Network(network={config.network}, netuid={config.netuid} staleness={self.network_staleness})')
        self.last_sync_time = c.time()

        # RESOLVE THE VOTING NETWORKS
        if 'local' in config.network:
            # local network does not need to be updated as it is atomically updated
            namespace = c.module('namespace').namespace(search=config.search, update=False)
        elif config.network in ['subspace', 'main', 'test']:
            if '.' in config.network:
                config.network, config.netuid = config.network.split('.')
            # convert the subnet to a netuid
            if isinstance(config.netuid, str):
                config.netuid = self.subspace.subnet2netuid(config.netuid)
            if '/' in config.network:
                _ , config.network = config.network.split('/')
            self.subspace = c.module('subspace')(network=config.network)
            config.network = 'subspace' + '/' + str(self.subspace.network)
            namespace = self.subspace.namespace(netuid=config.netuid, max_age=config.max_network_staleness)  
        else:
            raise Exception(f'Invalid network {config.network}')
        self.namespace = namespace
        self.namespace = {k: v for k, v in namespace.items() if self.filter_module(k)}
        self.n  = len(self.namespace)    
        self.name2address = self.namespace
        self.address2name = {k:v for v, k in namespace.items()}
        self.module2name = {v: k for k, v in self.namespace.items()}  
        self.config = config
        c.print(f'Synced network {config.network} with {self.n} modules', color='green')
        return self.network_info()
    
    sync = sync_network = set_network 

    @property
    def verbose(self):
        return self.config.verbose or self.config.debug

    def score(self, module: 'c.Module'):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        info = module.info()
        assert isinstance(info, dict) and not c.is_error(info), f'Info must be a dictionary, got {info}'
        return {'w': 1}
    
    def next_module(self):
        return c.choice(list(self.namespace.keys()))

    module2last_update = {}

    def get_module_path(self, module):
        if module in self.address2name:
            module = self.address2name[module]
        path = self.resolve_path(self.storage_path() +'/'+ module)
        return path

    def resolve_module_address(self, module):
        """
        reoslve the module address
        """
        if module in self.name2address:
            address = self.name2address[module]
        else:
            assert module in self.address2name, f"{module} is not found in {self.config.network}"
            address = module
            
        return address
    

    def check_info(self, info:dict) -> bool:
        return bool(isinstance(info, dict) and all([k in info for k in self.config.expected_info_keys]))



    def eval(self,  module:str, **kwargs):
        """
        The following evaluates a module sver
        """
        self.sync()
        info = {}
        try:
            info = {}
            # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
            address = self.resolve_module_address(module)
            path = self.get_module_path(module)
            module = c.connect(address, key=self.key)
            info = self.get_json(path, {})
            last_timestamp = info.get('timestamp', 0)
            info['staleness'] = c.time() -  last_timestamp
            if info['staleness'] < self.config.max_staleness:
                self.staleness_count += 1
                raise Exception({'module': info['name'], 
                    'msg': 'Module is too new and w', 
                    'staleness': info['staleness'], 
                    'max_staleness': self.config.max_staleness,
                    'timeleft': self.config.max_staleness - info['staleness'], 
                    })
            # is the info valid
            if not self.check_info(info):
                info = module.info(timeout=self.config.timeout_info)
            info['timestamp'] = c.timestamp() # the timestamp
            info['w'] = info.get('w', 0) # the weight from the module
            info['past_w'] = info['w'] # for calculating alpha
            info['path'] = path # path of saving the module
            self.last_sent = c.time()
            self.requests += 1
            response = self.score(module, **kwargs)
            response = self.process_response(response=response, info=info)
        except Exception as e:
            response = c.detailed_error(e)
            response['w'] = 0
            response['name'] = info.get('name', module)
            self.errors += 1
            self.last_error  = c.time() # the last time an error occured

        return response


    def check_response(self, response): 
        if type(response) in [int, float, bool]:
            # if the response is a number, we want to convert it to a dict
            response = {'w': float(response)}
        elif type(response) == dict:
            response = response
        else:
            raise Exception(f'Response must be a number or a boolean, got {response}')
        
        if not type(response['w']) in [int, float]:
            raise f'Response weight must be a number, got {response["w"]} with result : {response}'
        
    def process_response(self, response:dict, info:dict ):
        """
        Process the response from the score_module
        params:
            response
        """
        self.check_response(response)
        info.update(response)
        info['latency'] = c.round(c.time() - info['timestamp'], 3)
        info['w'] = info['w']  * self.config.alpha + info['past_w'] * (1 - self.config.alpha)
        info['history'] = info.get('history', []) + [{'w': info['w'], 'timestamp': info['timestamp']}]
        #  have a minimum weight to save storage of stale modules
        if info['w'] > self.config.min_leaderboard_weight:
            self.put_json(info['path'], info)
        self.successes += 1
        self.last_success = c.time()
        info['staleness'] = c.round(c.time() - info.get('timestamp', 0), 3)
        return info

    
    eval_module = eval

    def default_storage_path(self):
        network = self.config.network
        if 'subspace' in network:
            path = f'{network}.{self.config.netuid}'
        else:
            path = network
        return path
        
    def storage_path(self):
        # the set storage path in config.path is set, then the modules are saved in that directory
        storage_path =  self.config.get('storage_path', self.default_storage_path())
        storage_path = self.resolve_path(storage_path)
        self.config.storage_path = storage_path
        return self.config.storage_path

        
    def vote_info(self):
        try:
            if not self.is_voting_network:
                return {'success': False, 
                        'msg': 'Not a voting network' , 
                        'network': self.config.network , 
                        'voting_networks': self.voting_networks ,}
            votes = self.calculate_votes()
        except Exception as e:
            votes = {'uids': [], 'weights': []}
            c.print(c.detailed_error(e))
        return {
            'num_uids': len(votes.get('uids', [])),
            'staleness': self.vote_staleness,
            'key': self.key.ss58_address,
            'network': self.config.network,
        }
    
    def calculate_votes(self, **kwargs):
        leaderboard =  self.leaderboard(keys=['name', 'w', 'staleness','latency', 'key'],   to_dict=True)
        assert len(leaderboard) > 0
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        key2uid = self.subspace.key2uid(**kwargs) if hasattr(self, 'subspace') else {}
        for info in leaderboard:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if 'key' in info and info['w'] >= 0:
                if info['key'] in key2uid:
                    votes['keys'] += [info['key']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid.get(info['key'], -1)]
        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes

    votes = calculate_votes
    
    @property
    def votes_path(self):
        return self.storage_path() + f'/votes'

    def vote(self,**kwargs):
        votes =self.calculate_votes() 
        return self.subspace.set_weights(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                            weights=votes['weights'], 
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
                    keys = ['name', 'w',  'staleness', 'latency',  'address', 'staleness'],
                    max_age = None,
                    network = None,
                    ascending = True,
                    by = 'w',
                    to_dict = False,
                    n = None,
                    page = None,
                    **kwargs
                    ):
        max_age = max_age or self.config.max_leaderboard_age
        paths = self.module_paths()
        df = []
        # chunk the jobs into batches
        for path in paths:
            r = self.get(path, {},  max_age=max_age)
            if isinstance(r, dict) and 'key' and  r.get('w', 0) > self.config.min_leaderboard_weight  :
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
        if isinstance(by, str):
            by = [by]
        df = df.sort_values(by=by, ascending=ascending)

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
    
    def module_paths(self):
        paths = self.ls(self.storage_path())
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
            'staleness': self.network_staleness,

        }

    def run_info(self):
        return {
            'network': self.network_info(),
            'epoch': self.epoch_info() ,
            'vote': self.vote_info(),
          

            }
    

    @classmethod
    def from_module(cls, 
                   module,
                   config=None, 
                   functions = ['eval_module', 'score_module', 'eval', 'leaderboard', 'run_epoch'],
                   **kwargs):
        
        module = c.resolve_module(module)
        vali = cls(module=module, config=config, **kwargs)
        for fn in functions:
            setattr(module, fn, getattr(vali, fn))
        return module
    

    @classmethod
    def from_function(cls, 
                   function,
                   config=None, 
                   functions = ['eval_module', 'score_module', 'eval', 'leaderboard', 'run_epoch'],
                   **kwargs):
        vali = cls(config=config, **kwargs)
        for fn in functions:
            setattr(vali, fn, function)
        return vali

    def print_header(self):

        module_path = self.module_name()
        buffer='='*40
        c.print(buffer*2)
        c.print(buffer + module_path +  ' ' + buffer[len(module_path):])
        c.print(buffer*2)
    

Vali.run(__name__)
