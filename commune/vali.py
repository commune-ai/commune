
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):
    whitelist = ['eval_module', 'score', 'eval', 'leaderboard']
    voting_networks = ['bittensor', 'commune']
    def __init__(self,
                    network= 'local', # for local subspace:test or test # for testnet subspace:main or main # for mainnet
                    netuid = 0, # (NOT LOCAL) the subnetwork uid or the netuid. This is a unique identifier for the subnetwork 
                    search=  None, # (OPTIONAL) the search string for the network 
                    max_network_age=  10, # the maximum staleness of the network # LOGGING
                    verbose=  True, # the verbose mode for the worker # EPOCH
                    batch_size= 64,
                    queue_size=  128,
                    max_workers=  None ,
                    score = None, #EVAL
                    path= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                    alpha= 1.0, # alpha for score
                    min_leaderboard_weight=  0, # the minimum weight of the leaderboard
                    period =  3, # the interval for the run loop to run
                    tempo : int = None, # also period
                    run_loop= True, # This is the key that we need to change to false
                    vote_interval= 100, # the number of iterations to wait before voting
                    module = None,
                    timeout= 10, # timeout per evaluation of the module
                    timeout_info= 4, # (OPTIONAL) the timeout for the info worker
                    miner= False , # converts from a validator to a miner
                    update=False,
                    key = None,
                 **kwargs):
        period = period or tempo 
        if miner:
            run_loop = False

        config = self.set_config(locals())
        config = c.dict2munch({**Vali.config(), **config})
        self.config = config
        self.epochs = 0
        self.network_time = 0
        self.start_time = c.time() # the start time of the validator
        self.executor = c.module('executor')(max_workers=self.config.max_workers,  maxsize=self.config.queue_size)
        self.results = [] # the results of the evaluations
        self.futures = [] # the futures for the executor
        self.key = c.get_key(key or self.module_name())
        self.set_score(score)
        self.sync_network(update=update)
        if self.config.run_loop:
            c.thread(self.run_loop)


    init_vali = __init__

    def score(self, module):
        return 'name' in module.info()
    
    def score2(self, module):
        return float('name' in module.info())/3

    def set_score(self, score: Union[Callable, str] ):
        """
        Set the score function for the validator
        """
        if score == None:
            score = self.score
        elif c.object_exists(score):
            pass
        elif isinstance(score, str):
            if hasattr(self, score):
                score = getattr(self, score)
            else:
                score = c.get_fn(score)
        assert callable(score), f'{score} is not callable'
        setattr(self, 'score', score )
        return {'success': True, 'msg': 'Set score function', 'score': self.score.__name__}
    @property
    def lifetime(self):
        return c.time() - self.start_time
    
    @property
    def is_voting_network(self):
        return any([v in self.config.network for v in self.voting_networks])

    def run_step(self):
        """
        The following runs a step in the validation loop
        """
        self.sync_network()
        self.epoch()
        if self.is_voting_network and self.vote_staleness > self.config.vote_interval:
            c.print('Voting', color='cyan')
            c.print(self.vote())
        c.print(f'Epoch {self.epochs} with {self.n} modules', color='yellow')
        c.print(self.leaderboard())

    def run_loop(self):
        """
        The run loop is a backgroun loop that runs to do two checks
        - network: check the staleness of the network to resync it 
        - workers: check the staleness of the last success to restart the workers 
        - voting: check the staleness of the last vote to vote (if it is a voting network)
        
        """
        # start the workers

        while True:
            # count down the staleness of the last success
            from tqdm import tqdm
            # reverse the progress bar
            desc = f'Next Epoch ({self.epochs})'
            for i in tqdm(range(self.config.period), desc=desc, position=0, leave=True):
                c.sleep(1)
            try:
                self.run_step()
            except Exception as e:
                c.print(c.detailed_error(e))

    def age(self):
        return c.time() - self.start_time

    def get_next_result(self, futures=None):
        futures = futures or self.futures
        try:
            for future in c.as_completed(futures, timeout=self.config.timeout):
                futures.remove(future) 
                result = future.result()
                result['w'] = result.get('w', 0)
                did_score_bool = bool(result['w'] > 0)
                emoji =  '🟢' if did_score_bool else '🔴'
                if did_score_bool:
                    keys = ['w', 'name', 'address', 'latency']
                else:
                    keys = list(result.keys())
                result = {k: result.get(k, None) for k in keys if k in result}
                msg = ' '.join([f'{k}={result[k]}' for k in result])
                msg = f'RESULT({msg})'
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
        """
        The following runs an epoch for the validator
        
        """
        if self.epochs > 0:
            self.sync_network()
        self.epochs += 1
        module_addresses = c.shuffle(list(self.namespace.values()))
        c.print(f'Epoch {self.epochs} with {self.n} modules', color='yellow')
        batch_size = min(self.config.batch_size, len(module_addresses)//4)            
        results = []
        for module_address in module_addresses:
            if not self.executor.is_full:
                self.futures.append(self.executor.submit(self.eval, [module_address], timeout=self.config.timeout))
            if len(self.futures) >= batch_size:
                results.append(self.get_next_result(self.futures))
        while len(self.futures) > 0:
            results.append(self.get_next_result())
        results = [r for r in results if r.get('w', 0) > 0]
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
        return c.time() - self.network_time

    def filter_module(self, module:str, search=None):
        search = search or self.config.search
        if ',' in str(search):
            search_list = search.split(',')
        else:
            search_list = [search]
        return all([s == None or s in module  for s in search_list ])

    def sync_network(self,  
                     network:str=None, 
                      netuid:int=None,
                      search = None, 
                      update = False,
                      max_age=None):
        config = self.config
        network = network or config.network
        netuid =  netuid or config.netuid
        search = search or config.search
        max_age = max_age or config.max_network_age
        if update:
            max_age = 0
        if self.network_staleness < max_age:
            return {'msg': 'Alredy Synced network Within Interval', 
                    'staleness': self.network_staleness, 
                    'max_network_age': self.config.max_network_age,
                    'network': network, 
                    'netuid': netuid, 
                    'n': self.n,
                    'search': search,
                    }
        self.network_time = c.time()
        # RESOLVE THE VOTING NETWORKS
        if 'local' in network:
            # local network does not need to be updated as it is atomically updated
            namespace = c.get_namespace(search=search, update=1, max_age=max_age)
        elif 'subspace' in network:
            # the network is a voting network
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
            namespace = self.subspace.namespace(netuid=netuid, update=1)
        namespace = {k: v for k, v in namespace.items() if self.filter_module(k)}
        self.namespace = namespace
        self.n  = len(self.namespace)    
        config.network = network
        config.netuid = netuid
        self.config = config
        c.print(f'Network(network={config.network}, netuid={config.netuid} n=self.n)')
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
        return bool(isinstance(info, dict) and all([k in info for k in  ['w', 'address', 'name', 'key']]))

    def eval(self,  module:str, **kwargs):
        """
        The following evaluates a module sver
        """
        try:
            info = {}
            # RESOLVE THE NAME OF THE ADDRESS IF IT IS NOT A NAME
            path = self.resolve_path(self.path +'/'+ module)
            address = self.namespace.get(module, module)
            module_client = c.connect(address, key=self.key)
            info = self.get_json(path, {})
            last_timestamp = info.get('timestamp', 0)
            info['staleness'] = c.time() -  last_timestamp
            if not self.check_info(info):
                info = module_client.info(timeout=self.config.timeout_info)
            info['timestamp'] = c.timestamp() # the timestamp
            previous_w = info.get('w', 0)
            response = self.score(module_client, **kwargs)
            if type(response) in [int, float, bool]:
                response = {'w': response}
            response['w'] = float(response.get('w', 0))
            info.update(response)
            info['latency'] = c.round(c.time() - info['timestamp'], 3)
            alpha = self.config.alpha
            info['w'] = info['w']  * alpha + previous_w * (1 - alpha)
            if response['w'] > self.config.min_leaderboard_weight:
                self.put_json(path, info)
        except Exception as e:
            raise e
            response = c.detailed_error(e)
            response['w'] = 0
            response['name'] = info.get('name', module)
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
            votes = self.votes()
        except Exception as e:
            votes = {'uids': [], 'weights': []}
            c.print(c.detailed_error(e))
        return {
            'num_uids': len(votes.get('uids', [])),
            'staleness': self.vote_staleness,
            'key': self.key.ss58_address,
            'network': self.config.network,
        }
    
    def votes(self, **kwargs):
        leaderboard =  self.leaderboard(keys=['name', 'w', 'staleness','latency', 'key'],   to_dict=True)
        votes = {'modules': [], 'weights': []}
        for module in self.leaderboard().to_records():
            if module['w'] > 0:
                votes['modules'] += [module['key']]
                votes['weights'] += [module['w']]
        return votes



    
    @property
    def votes_path(self):
        return self.path + f'/votes'

    def vote(self,**kwargs):
        votes =self.votes() 
        return self.subspace.set_weights(modules=votes['modules'], # passing names as uids, to avoid slot conflicts
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
                    ascending = True,
                    by = 'w',
                    to_dict = False,
                    n = None,
                    page = None,
                    **kwargs
                    ):
        max_age = self.config.period
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
    
        
    @staticmethod
    def test( 
            n=2, 
                sleep_time=2, 
                timeout = 20,
                tag = 'vali_test_net',
                miner='module', 
                vali='vali', 
                storage_path = '/tmp/commune/vali_test',
                network='local'):
        
        test_miners = [f'{miner}::{tag}_{i}' for i in range(n)]
        test_vali = f'{vali}::{tag}'
        modules = test_miners
        search = tag
        for m in modules:
            c.kill(m) 
        for m in modules:
            c.print(c.serve(m))
        namespace = c.namespace(search=search)
        while len(namespace) < n:
            print(len(namespace))
            namespace = c.namespace(search=search)
        leaderboard = Vali.run_epoch(network=network, search=search, path=storage_path)
        assert len(leaderboard) == n, f'Leaderboard not updated {leaderboard}'
        for miner in modules:
            c.print(c.kill(miner))
        return {'success': True, 'msg': 'subnet test passed'}

if __name__ == '__main__':
    Vali.run()
