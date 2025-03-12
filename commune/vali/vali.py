
import commune as c
import os
import pandas as pd
from typing import *

class Vali:
    endpoints = ['score', 'scoreboard']
    epoch_time = 0
    vote_time = 0 # the time of the last vote (for voting networks)
    epochs = 0 # the number of epochs
    subnet = None
    network = 'server'
    tempo = 10

    def __init__(self,
                    network= 'local', # for local chain:test or test # for testnet chain:main or main # for mainnet
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 128, # the batch size of the most parallel tasks
                    score : Union['callable', int]= None, # score function
                    key : str = None,
                    tempo : int = 10 , 
                    timeout : int = 3, # timeout per evaluation of the module
                    update : bool =True, # update during the first epoch
                    run_loop : bool = True, # This is the key that we need to change to false
                    verbose: bool = True, # print verbose output
                    path : str= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                 **kwargs):     
        self.timeout = timeout
        self.batch_size = batch_size
        self.verbose = verbose
        self.key = c.get_key(key or self.module_name())
        self.set_network(network=network, tempo=tempo,  search=search,  path=path, update=update)
        self.set_score(score)
        if run_loop:
            c.thread(self.run_loop) if run_loop else ''

    init_vali = __init__

    @classmethod
    def resolve_path(cls, path):
        return c.storage_path + f'/vali/{path}'
    def set_network(self, 
                    network:str = None, 
                    tempo:int=None, 
                    search:str=None, 
                    path:str=None, 
                    update = False):
        self.network = network or self.network
        self.tempo = tempo or self.tempo
        if '/' in self.network:
            self.network, self.subnet = network.split('/')
        self.path = self.resolve_path((self.network + '/' + self.subnet) if self.subnet else self.network)
        self.netmod = c.module(self.network)() 
        self.params = self.netmod.params(subnet=self.subnet, max_age=self.tempo)
        self.modules = self.netmod.modules(subnet=self.subnet, max_age=self.tempo)
        if search:
            self.modules = [m for m in self.modules if any(str(search) in str(v) for v in m.values())]
        return self.params
    
    def score(self, module):
        return int('name' in module.info())
    
    def set_score(self, score: Union['callable', int]= None):
        if callable(score):
            setattr(self, 'score', score )
        assert callable(self.score), f'SCORE NOT SET {self.score}'
        return {'success': True, 'msg': 'Score function set'}

    def run_loop(self, step_time=2):
        while True:

            # wait until the next epoch
            seconds_until_epoch = int(self.epoch_time + self.tempo - c.time())
            if seconds_until_epoch > 0:
                progress = c.tqdm(total=seconds_until_epoch, desc='Time Until Next Progress')
                for i in range(seconds_until_epoch):
                    progress.update(step_time)
                    c.sleep(step_time)
            try:
                c.print(c.df(self.epoch()))
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')

    def score_module(self,  module:dict, **kwargs):
        t0 = c.time() # the timestamp
        if isinstance(module, str):
            module = {'url': module}
        client = c.client(module['url'], key=self.key)
        score = 0 
        try:
            score = self.score(client, **kwargs)
        except Exception as e:
            module['error'] = c.detailed_error(e)
            if self.verbose:
                print(f'ERROR({module["error"]})', color='red')
        module['score'] = score
        module['time'] = t0
        module['duration'] = c.time() - module['time']
        module['path'] = self.path +'/'+ module['key'] + '.json'
        module['proof'] = c.sign(module, key=self.key, mode='str')
        if module['score'] > 0:
            c.put_json(module['path'], module)
        return module

    def score_batch(self, modules: List[dict]):
        results = []
        try:
            futures = [c.submit(self.score_module, [m], timeout=self.timeout) for m in modules]   
            results = c.wait(futures, timeout=self.timeout)
        except Exception as e:
            c.print(f'ERROR({c.detailed_error(e)})', color='red')
        return results

    def epoch(self, **kwargs):
        self.set_network(**kwargs)
        n = len(self.modules)
        batches = [self.modules[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batches)
        c.print(f'Epoch(network={self.network} epoch={self.epochs} batch_size={self.batch_size} n={n} num_batches={num_batches})', color='yellow')
        progress = c.tqdm(total=len(batches), desc='Evaluating Modules')
        results = []
        for i, batch in enumerate(batches):
            c.print(f'Batch(i={i}/{num_batches} batch_size={len(batch)})', color='yellow')
            results.extend(self.score_batch(batch))
            progress.update(1)
        self.epochs += 1
        self.epoch_time = c.time()
        self.vote(results)
        for r in results: 
            r.pop('schema', None)
            r.pop('path')
        return results
    
    @property
    def votes_path(self):
        return self.path + f'/votes'


    @property
    def vote_staleness(self):
        return c.time() - self.vote_time

    def vote(self, results):
        if not bool(hasattr(self.netmod, 'vote')) :
            return {'success': False, 'msg': f'NOT VOTING NETWORK({self.network})'}
        if self.vote_staleness < self.tempo:
            return {'success': False, 'msg': f'Vote is too soon {self.vote_staleness}'}
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        assert all('score' in r for r in results), f'No score in results {results}'
        assert all('key' in r for r in results), f'No key in results {results}'

        return self.netmod.vote(
                    modules=[m['key'] for m in modules], 
                    weights=[m['score'] for m in modules],  
                    key=self.key, 
                    subnet=self.subnet
                    )
    
    def scoreboard(self,
                    keys = ['name', 'score', 'duration',  'url', 'key'],
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
        if to_dict:
            return df.to_dict(orient='records')
        if len(df) > page_size:
            pages = len(df)//page_size
            page = page or 0
            df = df[page*page_size:(page+1)*page_size]
        return df

    def module_paths(self):
        return c.ls(self.path) # fam
    
    @classmethod
    def run_epoch(cls, network='local', **kwargs):
        kwargs['run_loop'] = False
        return  cls(network=network,**kwargs).epoch()
    
    def refresh_scoreboard(self):
        path = self.path
        c.rm(path)
        return {'success': True, 'msg': 'Leaderboard removed', 'path': path}
