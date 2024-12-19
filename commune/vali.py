
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):
    endpoints = ['score', 'scoreboard']
    voting_networks = ['bittensor', 'subspace']
    networks = ['local'] + voting_networks
    epoch_time = 0
    vote_time = 0
    vote_staleness = 0
    epochs = 0

    def __init__(self,
                    network= 'local', # for local subspace:test or test # for testnet subspace:main or main # for mainnet
                    subnet : Optional[Union[str, int]] = None, # (OPTIONAL) the name of the subnetwork 
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 128, # the batch size of the most parallel tasks
                    max_workers : Optional[int]=  None , # the number of parallel workers in the executor
                    score : Union['callable', int]= None, # score function
                    key : str = None,
                    path : str= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                    tempo : int = None , 
                    timeout : int = 3, # timeout per evaluation of the module
                    update : bool =False, # update during the first epoch
                    run_loop : bool = True, # This is the key that we need to change to false
                 **kwargs):
        self.set_executor(max_workers=max_workers, batch_size=batch_size, timeout=timeout)
        self.set_network(network=network, subnet=subnet, tempo=tempo, search=search, path=path,  score=score, update=update)
        self.set_key(key)
        if run_loop:
            c.thread(self.run_loop)
    init_vali = __init__


    def set_key(self, key):
        self.key = key or c.get_key()
        return {'success': True, 'msg': 'Key set', 'key': self.key}

    def set_executor(self, max_workers:int, batch_size:int, timeout:int):
        self.timeout = timeout or 3
        self.max_workers = max_workers or c.cpu_count() * 5
        self.batch_size = batch_size or 128
        self.executor = c.module('executor')(max_workers=self.max_workers,  maxsize=self.batch_size)
        return {'success': True, 'msg': 'Executor set', 'max_workers': self.max_workers, 'batch_size': self.batch_size, 'timeout': self.timeout}
    
    def set_network(self, network:str, 
                    subnet:str=None, 
                    tempo:int=60, 
                    search:str=None, 
                    path:str=None, 
                    score = None,
                    update=False):
    

        if not network in self.networks and '/' not in network:
            network = f'subspace/{network}'
        [network, subnet] = network.split('/') if '/' in network else [network, subnet]
        self.subnet = subnet 
        self.network = network
        self.network_module = c.module(self.network)() 
        self.tempo = tempo
        self.search = search
        self.path = os.path.abspath(path or self.resolve_path(f'{network}/{subnet}'))
        self.set_score(score)
        self.sync(update=update)


    def score(self, module):
        return int('name' in module.info())
    
    def set_score(self, score):
        if callable(score):
            setattr(self, 'score', score )
        assert callable(self.score), f'SCORE NOT SET {self.score}'
        return {'success': True, 'msg': 'Score function set'}

    @property
    def is_voting_network(self):
        return any([v in self.network for v in self.voting_networks])
    
    def run_loop(self):
        while True:
            try:
                self.epoch()
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')
    @property
    def nex_epoch(self):
        return int(self.epoch_time + self.tempo - c.time())
    
    futures = []
    results = []

    def epoch(self):
        futures = []
        self.results = []
        next_epoch = self.nex_epoch
        progress = c.tqdm(total=next_epoch, desc='Next Epoch')
        for _ in  range(next_epoch):
            progress.update(1)
            c.sleep(1)
        self.sync()
        c.print(f'EPOCH(network={self.network} epoch={self.epochs} n={self.n})', color='yellow')
        progress = c.tqdm(total=self.n, desc='Evaluating Modules')
        # return self.modules
        n = len(self.modules)
        
        for i, module in enumerate(self.modules):
            module["i"] = i
            c.print(f'EVAL(i={i}/{n} key={module["key"]} name={module["name"]})', color='yellow')
            if len(futures) < self.batch_size:
                futures.append(self.executor.submit(self.score_module, [module], timeout=self.timeout))
            else: 
                self.results.append(self.next_result(futures))
            progress.update(1)
        while len(futures) > 0:
            self.results.append(self.next_result(futures))
        self.results = [r for r in self.results if r.get('score', 0) > 0]
        self.epochs += 1
        self.epoch_time = c.time()
        c.print(self.vote())
        print(self.scoreboard())
        return self.results
    
    def sync(self, update = False):
        max_age =  0 if update else (self.tempo or 60)
        self.modules = self.network_module.modules(subnet=self.subnet, max_age=max_age)
        self.params = self.network_module.params(subnet=self.subnet, max_age=max_age)
        self.tempo =  self.tempo or (self.params['tempo'] * self.network_module.block_time)//2
        if self.search != None:
            self.modules = [m for m in self.modules if self.search in m['name']]
        self.n  = len(self.modules)  
        self.network_info = {'n': self.n, 'network': self.network ,  'subnet': self.subnet, 'params': self.params}
        c.print(f'<Network({self.network_info})')
        return self.network_info

    def score_module(self,  module:dict, **kwargs):
        module['time'] = c.time() # the timestamp
        module['score'] = self.score(c.connect(module['address'], key=self.key), **kwargs)
        module['latency'] = c.time() - module['time']
        if module['score'] > 0:
            module_path = self.path +'/'+ module['key']
            c.put_json(module_path, module)
        return module
    
    @property
    def votes_path(self):
        return self.path + f'/votes'

    def vote(self, submit_async=True, **kwargs):

        if not self.is_voting_network :
            return {'success': False, 'msg': f'NOT VOTING NETWORK({self.network})'}
        self.vote_staleness = c.time() - self.vote_time
        if self.vote_staleness < self.tempo:
            return {'success': False, 'msg': f'Vote is too soon {self.vote_staleness}'}
        fn_obj = self.network_module.vote
        params = dict(modules=[], weights=[], key=self.key, subnet=self.subnet)
        if len(self.results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        for m in self.results:
            params['modules'].append(m['key'])
            params['weights'].append(m['score'])
        if submit_async:
            return c.submit(fn_obj, params)
        else:
            return self.network_module.vote(**params)
    
    set_weights = vote 

    def module_info(self, **kwargs):
        if hasattr(self, 'network_module'):
            return self.network_module.module_info(self.key.ss58_address, subnet=self.subnet, **kwargs)
        else:
            return self.info()
    
    def scoreboard(self,
                    keys = ['name', 'score', 'latency',  'address', 'key'],
                    ascending = True,
                    by = 'score',
                    to_dict = False,
                    n = None,
                    page = None,
                    **kwargs
                    ):
        max_age = self.tempo
        df = []
        # chunk the jobs into batches
        for path in self.module_paths():
            r = self.get(path, {},  max_age=max_age)
            if isinstance(r, dict) and 'key' and  r.get('score', 0) > 0  :
                df += [{k: r.get(k, None) for k in keys}]
            else :
                self.rm(path)

        df = c.df(df) 
        if len(df) > 0:
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

    def module_paths(self):
        paths = self.ls(self.path)
        return paths
    
    @classmethod
    def run_epoch(cls, network='local', run_loop=False, update=False, **kwargs):
        return  cls(network=network, run_loop=run_loop, update=update, **kwargs).epoch()

    def next_result(self, futures:list, features=['score', 'name', 'key']):
        try:
            for future in c.as_completed(futures, timeout=self.timeout):
                    futures.remove(future) 
                    result = future.result()
                    if all([f in result for f in features]):
                        v_result = {f: result[f] for f in features}
                    
                        c.print(f'RESULT({v_result})', color='red')
                        return result
                    else:
                        v_result = {f: result[f] for f in result if f not in ['success']}
                        c.print(f'ERROR({result["error"]})', color='red')

        except Exception as e:
            result = c.detailed_error(e)
            result.pop('success')
            c.print(f'ERROR({result})', color='red')
        return result
    
    @staticmethod
    def test(  
             n=2, 
             tag = 'vali_test_net',  
             miner='module', 
             trials = 5,
             tempo = 4,
             update=True,
             path = '/tmp/commune/vali_test',
             network='local'
             ):
        test_miners = [f'{miner}::{tag}{i}' for i in range(n)]
        modules = test_miners
        search = tag
        assert len(modules) == n, f'Number of miners not equal to n {len(modules)} != {n}'
        for m in modules:
            c.serve(m)
        namespace = c.namespace()
        for m in modules:
            assert m in namespace, f'Miner not in namespace {m}'
        vali = Vali(network=network, search=search, path=path, update=update, tempo=tempo, run_loop=False)
        print(vali.modules)
        scoreboard = []
        while len(scoreboard) < n:
            c.sleep(1)
            scoreboard = vali.epoch()
            trials -= 1
            assert trials > 0, f'Trials exhausted {trials}'
        for miner in modules:
            c.print(c.kill(miner))
        return {'success': True, 'msg': 'subnet test passed'}
    
    def refresh_scoreboard(self):
        path = self.path
        c.rm(path)
        return {'success': True, 'msg': 'Leaderboard removed', 'path': path}