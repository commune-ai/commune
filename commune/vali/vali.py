
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):

    whitelist = ['eval_module', 'score', 'eval', 'leaderboard']
    voting_networks = ['bittensor', 'commune']

    def __init__(self,
                    # NETWORK
                    network= 'local', # for local subspace:test or test # for testnet subspace:main or main # for mainnet
                    netuid = 0, # (NOT LOCAL) the subnetwork uid or the netuid. This is a unique identifier for the subnetwork 
                    search=  None, # (OPTIONAL) the search string for the network 
                    max_network_staleness=  10, # the maximum staleness of the network
                    # LOGGING
                    verbose=  True, # the verbose mode for the worker
                    # EPOCH
                    batch_size= 64,
                    queue_size=  128,
                    max_workers=  None ,
                    score_fn = None,
                    #EVAL
                    path= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                    alpha= 1.0, # alpha for score
                    timeout= 10, # timeout per evaluation of the module
                    max_staleness= 0, # the maximum staleness of the worker
                    epoch_time=  3600, # the maximum age of the leaderboard befor it is refreshed
                    min_leaderboard_weight=  0, # the minimum weight of the leaderboard
                    run_step_interval =  3, # the interval for the run loop to run
                    run_loop= True, # This is the key that we need to change to false
                    vote_interval= 100, # the number of iterations to wait before voting
                    module = None,
                    timeout_info= 4, # (OPTIONAL) the timeout for the info worker
                    miner= False , # converts from a validator to a miner
                    update=False,
                 **kwargs):
        max_workers = max_workers or batch_size
        config = self.set_config(locals())
        config = c.dict2munch({**Vali.config(), **config})
        self.config = config
        if update:
            self.config.max_staleness = 0
        self.sync()
        # start the run loop
        if self.config.run_loop:
            c.thread(self.run_loop)
    init_vali = __init__
        


    def score(self, module):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        a = c.random_int()
        b = c.random_int()
        expected_output = b
        module.put_item(str(a),b)
        output = module.get_item(str(a))
        if output == expected_output:
            return 1
        return 0

        
    def set_score_fn(self, score_fn: Union[Callable, str]):
        """
        Set the score function for the validator
        """
        module = module or self 
        if isinstance(score_fn, str):
            score_fn = c.get_fn(score_fn)
        assert callable(score_fn)
        self.score = getattr(self, score_fn)
        return {'success': True, 'msg': 'Set score function', 'score_fn': self.score.__name__}

    def init_state(self):
        self.executor = c.module('executor.thread')(max_workers=self.config.max_workers,  maxsize=self.config.queue_size)
        self.futures = []
        self.state = c.dict2munch(dict(
            requests = 0,
            last_start_time = 0,
            errors  = 0, 
            successes = 0, 
            epochs = 0, 
            last_sync_time = 0,
            last_error = 0,
            last_sent = 0,
            last_success = 0,
            start_time = c.time(),
            results = [],
            futures = [],
        ))


    @property
    def sent_staleness(self):
        return c.time()  - self.state.last_sent

    @property
    def success_staleness(self):
        return c.time() - self.state.last_success

    @property
    def lifetime(self):
        return c.time() - self.state.start_time
    
    @property
    def is_voting_network(self):
        return any([v in self.config.network for v in self.voting_networks])
    
    @property
    def last_start_staleness(self):
        return c.time() - self.last_start_time

    def run_step(self):
        """
        The following runs a step in the validation loop
        """
        self.epoch()
        if self.is_voting_network and self.vote_staleness > self.config.vote_interval:
            c.print('Voting', color='cyan')
            c.print(self.vote())
        c.print(f'Epoch {self.state.epochs} with {self.n} modules', color='yellow')
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

        while True:
            c.sleep(self.config.run_step_interval)
            try:
                self.run_step()
            except Exception as e:
                c.print(c.detailed_error(e))

    def age(self):
        return c.time() - self.state.start_time

    def wait_for_result(self):
        try:
            for future in c.as_completed(self.futures, timeout=self.config.timeout):
                self.futures.remove(future) 
                result = future.result()
                result['w'] = result.get('w', 0)

                did_score_bool = bool(result['w'] > 0)
                emoji =  '🟢' if did_score_bool else '🔴'
                if did_score_bool:
                    result = {k: result.get(k, None) for k in ['w',  'name', 'key', 'address'] if k in result}
                    msg = ' '.join([f'{k}={result[k]}' for k in result])
                    msg = f'SUCCESS({msg})'
                else:
                    result = {k: result.get(k, None) for k in ['w', 'name', 'key', 'address', 'msg'] if k in result}
                    msg = ' '.join([f'{k}={result[k]}' for k in result.keys()])
                    msg =  f'ERROR({msg})'
                break
        except Exception as e:
            emoji = '🔴'
            result = c.detailed_error(e)
            msg = f'Error({result})'
            
        c.print(emoji + msg + emoji, 
                color='cyan', 
                verbose=True)
        return result


    def cancel_futures(self):
        for f in self.futures:
            f.cancel()

    epoch2results = {}

    @classmethod
    def run_epoch(cls, network='local', vali=None, run_loop=False, update=1, **kwargs):
        if vali != None:
            cls = c.module(vali)
        self = cls(network=network, run_loop=run_loop, update=update, **kwargs)
        return self.epoch(df=1)

    def epoch(self, df=True):
        self.state.epochs += 1
        self.sync()
        module_addresses = c.shuffle(list(self.namespace.values()))
        c.print(f'Epoch {self.state.epochs} with {self.n} modules', color='yellow')
        batch_size = min(self.config.batch_size, len(module_addresses)//4)            
        results = []
        timeout = self.config.timeout
        self.current_epoch = self.state.epochs
        for module_address in module_addresses:
            if not self.executor.is_full:
                future = self.executor.submit(self.eval, {'module': module_address},timeout=timeout)
                self.futures.append(future)
            if len(self.futures) >= batch_size:
                results.append(self.wait_for_result())
        while len(self.futures) > 0:
            results.append(self.wait_for_result())
        results = [r for r in results if not c.is_error(r)]
        if df:
            if len(results) > 0 and 'w' in results[0]:
                results =  c.df(results)
                results = results.sort_values(by='w', ascending=False)
        return results
    
    
    @property
    def network_staleness(self) -> int:
        """
        The staleness of the network
        """
        return c.time() - self.state.last_sync_time

    def filter_module(self, module:str, search=None):
        search = search or self.config.search
        if ',' in str(search):
            search_list = search.split(',')
        else:
            search_list = [search]
        return all([s == None or s in module  for s in search_list ])

    
    def sync(self,  
                     network:str=None, 
                      netuid:int=None,
                      search = None, 
                      max_network_staleness=None):
        self.init_state()
        config = self.config
        network = network or config.network
        netuid =  netuid or config.netuid
        search = search or config.search
        max_network_staleness = max_network_staleness or config.max_network_staleness
        if self.network_staleness < max_network_staleness:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'staleness': self.network_staleness, 
                    'max_network_staleness': self.config.max_network_staleness,
                    'network': network, 
                    'netuid': netuid, 
                    'n': self.n,
                    'search': search,
                    }
        
        self.state.last_sync_time = c.time()
        # RESOLVE THE VOTING NETWORKS
        if 'local' in config.network:
            # local network does not need to be updated as it is atomically updated
            namespace = c.namespace(search=config.search, update=1)

        else:
            for sep in [':', '/']:
                if sep in config.network:
                    # subtensor{sep}test
                    if len(config.network.split(sep)) == 2:
                        _ , network = config.network.split(sep)
                    elif len(config.network.split(sep)) == 3:
                        _ , network, netuid = config.network.split(sep)
                        netuid = int(netuid)
                    break

                # the network is a voting network
                self.subspace = c.module('subspace')(network=network, netuid=netuid)
                namespace = self.subspace.namespace(netuid=config.netuid, update=1)


        namespace = {k: v for k, v in namespace.items() if self.filter_module(k)}
        self.name2address = {k:v for k, v in namespace.items()}
        self.address2name = {v: k for k, v in namespace.items()} 
        self.namespace = namespace
        self.n  = len(self.namespace)    
        config.network = network
        config.netuid = netuid
        self.config = config
        c.print(f'Network(network={config.network}, netuid={config.netuid} staleness={self.network_staleness})')
        self.network_state = {
            'network': network,
            'netuid': netuid,
            'n': self.n,
            'search': search,
            'namespace': namespace,
            
        }

        self.put_json(self.path + '/network', self.network_state)

        return 
    

    
    def next_module(self):
        return c.choice(list(self.namespace.keys()))

    module2last_update = {}
    
    def check_info(self, info:dict) -> bool:
        expected_info_keys =  ['w', 'address', 'name', 'key'] # the keys for the expected info function
        return bool(isinstance(info, dict) and all([k in info for k in expected_info_keys]))

    def eval(self,  module:str, **kwargs):
        """
        The following evaluates a module sver
        """
        alpha = self.config.alpha
        try:
            info = {}
            # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
            path = self.resolve_path(self.path +'/'+ module)
            address = self.namespace.get(module, module)
            module_client = c.connect(address, key=self.key)
            info = self.get_json(path, {})
            last_timestamp = info.get('timestamp', 0)
            info['staleness'] = c.time() -  last_timestamp
            if info['staleness'] < self.config.max_staleness:
                raise Exception({'module': info['name'], 
                    'msg': 'Too New', 
                    'staleness': info['staleness'], 
                    'max_staleness': self.config.max_staleness,
                    'timeleft': self.config.max_staleness - info['staleness'], 
                    })
            # is the info valid
            if not self.check_info(info):
                info = module_client.info(timeout=self.config.timeout_info)
            self.state.last_sent = c.time()
            self.state.requests += 1
            info['timestamp'] = c.timestamp() # the timestamp
            previous_w = info.get('w', 0)
            # SCORE 
            response = self.score(module_client, **kwargs)
            # PROCESS THE SCORE
            if type(response) in [int, float, bool]:
                # if the response is a number, we want to convert it to a dict
                response = {'w': response}
            response['w'] = float(response.get('w', 0))

            info.update(response)
            info['latency'] = c.round(c.time() - info['timestamp'], 3)
            info['w'] = info['w']  * alpha + previous_w * (1 - alpha)
            info['history'] = info.get('history', []) + [{'w': info['w'], 'timestamp': info['timestamp']}]
            #  have a minimum weight to save storage of stale modules
            self.state.successes += 1
            self.state.last_success = c.time()
            info['staleness'] = c.round(c.time() - info.get('timestamp', 0), 3)

            if response['w'] > self.config.min_leaderboard_weight:
                self.put_json(path, info)
                
        except Exception as e:
            response = c.detailed_error(e)
            response['w'] = 0
            response['name'] = info.get('name', module)
            self.errors += 1
            self.state.last_error  = c.time() # the last time an error occured
            info.update(response)
        
        return info
    
    eval_module = eval
      
    @property
    def path(self):
        # the set storage path in config.path is set, then the modules are saved in that directory
        default_path = f'{self.config.network}.{self.config.netuid}' if self.is_voting_network else self.config.network
        self.config.path = self.resolve_path(self.config.get('path', default_path))
        return self.config.path

    def vote_info(self):
        try:
            if not self.is_voting_network:
                return {'success': False, 
                        'msg': 'Not a voting network' , 
                        'network': self.config.network , 
                        'voting_networks': self.voting_networks}
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
        return self.path + f'/votes'

    def vote(self,**kwargs):
        votes =self.calculate_votes() 
        return self.subspace.set_weights(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                            weights=votes['weights'], 
                            key=self.key, 
                            network=self.config.network, 
                            netuid=self.config.netuid,
                            )
    
    set_weights = vote 

    def module_info(self, **kwargs):
        if hasattr(self, 'subspace'):
            return self.subspace.module_info(self.key.ss58_address, netuid=self.config.netuid, **kwargs)
        else:
            return {}
    
    def leaderboard(self,
                    keys = ['name', 'w',  'staleness', 'latency',  'address', 'staleness', 'key'],
                    max_age = None,
                    ascending = True,
                    by = 'w',
                    to_dict = False,
                    n = None,
                    page = None,
                    **kwargs
                    ):
        max_age = max_age or self.config.epoch_time
        paths = self.paths()
        df = []
        # chunk the jobs into batches
        for path in paths:
            r = self.get(path, {},  max_age=max_age)
            if isinstance(r, dict) and 'key' and  r.get('w', 0) > self.config.min_leaderboard_weight  :
                r['staleness'] = c.time() - r.get('timestamp', 0)
                if not self.filter_module(r.get('name', None)):
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

    def paths(self):
        paths = self.ls(self.path)
        return paths
    
    def refresh_leaderboard(self):
        path = self.path
        r = self.rm(path)
        df = self.leaderboard()
        assert len(df) == 0, f'Leaderboard not removed {df}'
        return {'success': True, 'msg': 'Leaderboard removed', 'path': path}
    
    refresh = refresh_leaderboard 
    
    def save_module_info(self, k:str, v:dict,):
        path = self.path + f'/{k}'
        self.put(path, v)

    @property
    def vote_staleness(self):
        try:
            if 'subspace' in self.config.network:
                return self.subspace.block - self.module_info()['last_update']
        except Exception as e:
            pass
        return 0
 

    @classmethod
    def from_module(cls, 
                   module,
                   config=None, 
                   functions = ['eval_module', 'score', 'eval', 'leaderboard', 'run_epoch'],
                   **kwargs):
        
        module = c.resolve_module(module)
        vali = cls(module=module, config=config, **kwargs)
        for fn in functions:
            setattr(module, fn, getattr(vali, fn))
        return module
    
    @classmethod
    def from_function(cls, 
                   function,
                   **kwargs):
        vali = cls( **kwargs)
        setattr(vali, 'score_module', function)
        return vali
    
 

Vali.run(__name__)
