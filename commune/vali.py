
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
    vote_staleness = 0 # the time since the last vote
    epochs = 0 # the number of epochs
    futures = [] # the futures for the parallel tasks
    results = [] # the results of the parallel tasks
    _clients = {} # the clients for the parallel tasks

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

        self.timeout = timeout or 3
        self.max_workers = max_workers or c.cpu_count() * 5
        self.batch_size = batch_size or 128
        self.executor = c.module('executor')(max_workers=self.max_workers,  maxsize=self.batch_size)
        self.set_key(key)
        self.set_network(network=network, subnet=subnet, tempo=tempo, search=search, path=path,  score=score, update=update)
        if run_loop:
            c.thread(self.run_loop)
    init_vali = __init__


    def set_key(self, key):
        self.key = c.get_key(key or self.module_name())
        return {'success': True, 'msg': 'Key set', 'key': self.key}

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
        self.path = os.path.abspath(path or self.resolve_path(f'{network}/{subnet}' if subnet else network))
        self.is_voting_network = any([v in self.network for v in self.voting_networks])
        self.set_score(score)
        self.sync(update=update)

    def score(self, module):
        print(module.info(), 'FAM')
        return int('name' in module.info())
    
    def set_score(self, score):
        if callable(score):
            setattr(self, 'score', score )
        assert callable(self.score), f'SCORE NOT SET {self.score}'
        return {'success': True, 'msg': 'Score function set'}

    def run_loop(self):
        while True:
            try:
                self.epoch()
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')
    @property
    def time_until_next_epoch(self):
        return int(self.epoch_time + self.tempo - c.time())


    def get_client(self, module:dict):
        if module['key'] in self._clients:
            client =  self._clients[module['key']]
        else:
            client =  c.connect(module['address'], key=self.key)
            self._clients[module['key']] = client
        return client
    
    def score_module(self,  module:dict, **kwargs):
        """
        module: dict
            name: str
            address: str
            key: str
            time: int
        """
        if isinstance(module, str):
            module = self.network_module.get_module(module)
        module['time'] = c.time() # the timestamp
        client = self.get_client(module)
        try:
            module['score'] = self.score(client, **kwargs)
        except Exception as e:
            module['score'] = 0
            module['error'] = c.detailed_error(e)
        module['latency'] = c.time() - module['time']
        module['path'] = self.path +'/'+ module['key']
        return module

    def score_modules(self, modules: List[dict]):
        module_results = []
        futures = [self.executor.submit(self.score_module, [m], timeout=self.timeout) for m in modules]   
        try:
            for f in c.as_completed(futures, timeout=self.timeout):
                m = f.result()
                print(m)
                if m.get('score', 0) > 0:
                    c.put_json(m['path'], m)
                    module_results.append(m)
        except Exception as e:
            c.print(f'ERROR({c.detailed_error(e)})', color='red', verbose=1)
        print(module_results)
        return module_results

    def epoch(self):
        next_epoch = self.time_until_next_epoch
        progress = c.tqdm(total=next_epoch, desc='Next Epoch')
        for _ in  range(next_epoch):
            progress.update(1)
            c.sleep(1)
        self.sync()
        c.print(f'Epoch(network={self.network} epoch={self.epochs} n={self.n} )', color='yellow')
        batches = [self.modules[i:i+self.batch_size] for i in range(0, self.n, self.batch_size)]
        progress = c.tqdm(total=len(batches), desc='Evaluating Modules')
        results = []
        for i, module_batch in enumerate(batches):
            print(f'Batch(i={i}/{len(batches)})')
            results += self.score_modules(module_batch)
            progress.update(1)
        self.epochs += 1
        self.epoch_time = c.time()
        print(self.scoreboard())
        self.vote(results)
        return results
    
    def sync(self, update = False):
        max_age =  0 if update else (self.tempo or 60)
        self.modules = self.network_module.modules(subnet=self.subnet, max_age=max_age)
        self.params = self.network_module.params(subnet=self.subnet, max_age=max_age)
        self.tempo =  self.tempo or (self.params['tempo'] * self.network_module.block_time)//2
        print(self.tempo)
        if self.search != None:
            self.modules = [m for m in self.modules if self.search in m['name']]
        self.n  = len(self.modules)  
        self.network_info = {'n': self.n, 'network': self.network ,  'subnet': self.subnet, 'params': self.params}
        c.print(f'<Network({self.network_info})')
        return self.network_info
    
    @property
    def votes_path(self):
        return self.path + f'/votes'

    def vote(self, results):
        if not self.is_voting_network :
            return {'success': False, 'msg': f'NOT VOTING NETWORK({self.network})'}
        if c.time() - self.vote_time < self.tempo:
            return {'success': False, 'msg': f'Vote is too soon {self.vote_staleness}'}
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        params = dict(modules=[],  
                      weights=[],  
                      key=self.key,
                      subnet=self.subnet)
        for m in results:
            if not isinstance(m, dict) or 'key' not in m:
                continue
            params['modules'].append(m['key'])
            params['weights'].append(m['score'])
        return self.network_module.vote(**params)
    
    
    def scoreboard(self,
                    keys = ['name', 'score', 'latency',  'address', 'key'],
                    ascending = True,
                    by = 'score',
                    to_dict = False,
                    page = None,
                    **kwargs
                    ):
        page_size = 1000
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
        # if to_dict is true, we return the dataframe as a list of dictionaries
        if to_dict:
            return df.to_dict(orient='records')
        if len(df) > page_size:
            pages = len(df)//page_size
            page = page or 0
            df = df[page*page_size:(page+1)*page_size]

        return df

    def module_paths(self):
        paths = self.ls(self.path)
        return paths
    
    @classmethod
    def run_epoch(cls, network='local', run_loop=False, update=False, **kwargs):
        return  cls(network=network, run_loop=run_loop, update=update, **kwargs).epoch()
    
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