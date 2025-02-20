
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):
    endpoints = ['score', 'scoreboard']
    epoch_time = 0
    vote_time = 0 # the time of the last vote (for voting networks)
    epochs = 0 # the number of epochs
    def __init__(self,
                    network= 'local', # for local subspace:test or test # for testnet subspace:main or main # for mainnet
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 128, # the batch size of the most parallel tasks
                    score : Union['callable', int]= None, # score function
                    key : str = None,
                    tempo : int = 60 , 
                    timeout : int = 3, # timeout per evaluation of the module
                    update : bool =False, # update during the first epoch
                    run_loop : bool = True, # This is the key that we need to change to false
                    path : str= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                 **kwargs):     
        self.timeout = timeout
        self.batch_size = batch_size
        self.set_key(key)
        self.set_network(network=network, tempo=tempo,  search=search,  path=path, update=update)
        self.set_score(score)
        c.thread(self.run_loop) if run_loop else ''
    init_vali = __init__


    @classmethod
    def resolve_path(cls, path):
        return c.storage_path + f'/vali/{path}'

    def set_score(self, score=None):
        if score == None:
            score = self.score
        if isinstance(score, str):
            score = c.get_fn(score)
        if callable(score):
            setattr(self, 'score', score )
        c.print(f'Score({self.score})')
        assert callable(self.score), f'SCORE NOT SET {self.score}'
        return {'success': True, 'msg': 'Score function set'}
    
    def set_key(self, key):
        self.key = c.get_key(key or self.module_name())
        return {'success': True, 'msg': 'Key set', 'key': self.key}

    def set_network(self, 
                    network:str = None, 
                    tempo:int=60, 
                    search:str=None, 
                    path:str=None, 
                    update = False):
        self.network = network or 'server'
        self.subnet = None
        if '/' in self.network:
            self.network, self.subnet = network.split('/')
        if self.subnet == None:
            self.path = os.path.abspath(path or self.resolve_path(f'{self.network}/{self.subnet}' if self.subnet else self.network))
        self.netmod = c.module(self.network)() 
        self.tempo = tempo or 60
        self.search = search
        self.sync_net()
        c.print(f'Network(net={self.network} path={self.path})')
        return {'success': True, 'msg': 'Network set', 'network': self.network, 'path': self.path}

    def sync_net(self, max_age=None, update=False):
        max_age = max_age or self.tempo
        self.params = self.netmod.params(subnet=self.subnet, max_age=max_age)
        self.modules = self._modules = self.netmod.modules(subnet=self.subnet, max_age=max_age)
        if self.search:
            self.modules = [m for m in self.modules if any(str(self.search) in str(v) for v in m.values())]
        return self.params
    
    def score(self, module):
        info = module.info()
        return int('name' in info)
    
    def set_score(self, score: Union['callable', int]= None):
        if callable(score):
            setattr(self, 'score', score )
        assert callable(self.score), f'SCORE NOT SET {self.score}'
        return {'success': True, 'msg': 'Score function set'}

    def run_loop(self):
        while True:
            if self.time_until_next_epoch > 0:
                progress = c.tqdm(total=self.time_until_next_epoch, desc='Time Until Next Progress')
                for i in range(self.time_until_next_epoch):
                    progress.update(1)
                    c.sleep(1)

            try:
                c.print(c.df(self.epoch()))
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')
    @property
    def time_until_next_epoch(self):
        return int(self.epoch_time + self.tempo - c.time())

    _clients = {}
    def get_client(self, module:dict) -> 'commune.Client':
        if isinstance(module, str):
            module = c.call(module, key=self.key)
            print(module)

        feature2type = {'name': str, 'url': str, 'key': str}
        for f, t in feature2type.items():
            assert f in module, f'Module missing {f}'
            assert isinstance(module[f], t), f'Module {f} is not {t} {module}'

        if module['key'] not in self._clients:
            self._clients[module['key']] =  c.client(module['url'], key=self.key)
        return  self._clients[module['key']]


    def score_module(self,  module:dict, **kwargs):
        client = self.get_client(module) # the client
        t0 = c.time() # the timestamp
        score = 0 
        try:
            score = self.score(client, **kwargs)
        except Exception as e:
            module['error'] = c.detailed_error(e)
        module['score'] = score
        module['time'] = t0
        module['latency'] = c.time() - module['time']
        module['path'] = self.path +'/'+ module['key'] + '.json'
        return module

    def score_batch(self, modules: List[dict]):
        try:
            results = []
            futures = [c.submit(self.score_module, [m], timeout=self.timeout) for m in modules]   
            for f in c.as_completed(futures, timeout=self.timeout):
                m = f.result()
                print(m)
                if m.get('score', 0) > 0:
                    c.put_json(m['path'], m)
                    results.append(m)
        except Exception as e:
            c.print(f'ERROR({c.detailed_error(e)})', color='red')
        return results

    def epoch(self):
        self.sync_net(update=1)
        n = len(self.modules)
        batches = [self.modules[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batches)
        c.print(f'Epoch(network={self.network} epoch={self.epochs} batch_size={self.batch_size} n={n} num_batches={num_batches})', color='yellow')
        progress = c.tqdm(total=len(batches), desc='Evaluating Modules')
        results = []
        for i, batch in enumerate(batches):
            c.print(f'Batch(i={i}/{num_batches} batch_size={len(batch)})', color='yellow')
            results += self.score_batch(batch)
            progress.update(1)
        self.epochs += 1
        self.epoch_time = c.time()
        self.vote(results)
        return results
    
    @property
    def votes_path(self):
        return self.path + f'/votes'

    def vote(self, results):
        voting_network = bool(hasattr(self.netmod, 'vote'))
        if not voting_network :
            return {'success': False, 'msg': f'NOT VOTING NETWORK({self.network})'}
        vote_staleness = c.time() - self.vote_time
        if vote_staleness < self.tempo:
            return {'success': False, 'msg': f'Vote is too soon {vote_staleness}'}
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        params = dict(modules=[], weights=[],  key=self.key, subnet=self.subnet)
        for m in results:
            if not isinstance(m, dict) or 'key' not in m:
                continue
            params['modules'].append(m['key'])
            params['weights'].append(m['score'])
        return self.netmod.vote(**params)
    
    def scoreboard(self,
                    keys = ['name', 'score', 'latency',  'url', 'key'],
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
            Vali  = c.module('vali')
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