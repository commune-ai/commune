
import commune as c
import os
import pandas as pd
from typing import *

class Vali(c.Module):
    endpoints = ['score', 'scoreboard']
    voting_networks = ['bittensor', 'commune', 'subspace']
    networks = ['local'] + voting_networks

    def __init__(self,
                    network= 'local', # for local subspace:test or test # for testnet subspace:main or main # for mainnet
                    subnet : Optional[Union[str, int]] = None, # (OPTIONAL) the name of the subnetwork 
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    verbose : bool =  True, # the verbose mode for the worker # EPOCH
                    batch_size : int = 16, # the batch size of the most parallel tasks
                    max_workers : Optional[int]=  None , # the number of parallel workers in the executor
                    score : Union['callable', int]= None, # score function
                    path= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                    min_score=  0, # the minimum weight of the scoreboard
                    run_loop= True, # This is the key that we need to change to false
                    test= False, # the test mode for the validator
                    tempo = None , 
                    timeout= 10, # timeout per evaluation of the module
                    update=False, # update during the first epoch
                    key = None,
                 **kwargs):
        if not network in self.networks:
            network = f'subspace/{network}'
        self.epochs = 0
        self.max_workers = max_workers or c.cpu_count() * 5
        self.batch_size = batch_size
        self.min_score = min_score
        self.timeout = timeout
        self.test = test
        self.verbose = verbose
        self.search = search
        if callable(score):
            setattr(self, 'score', score )
        self.set_key(key)
        network, subnet = network.split('/') if '/' in network else [network, subnet]
        self.subnet = subnet 
        self.network = network
        self.tempo = tempo
        self.path = self.resolve_path( path or f'{network}/{subnet}')
        self.sync(update=update)
        if run_loop:
            c.thread(self.run_loop)

    init_vali = __init__
    
    def score(self, module):
        return int('name' in module.info())


    @property
    def is_voting_network(self):
        return any([v in self.network for v in self.voting_networks])
    
    def run_loop(self):
        while True:
            try:
                self.epoch()
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')

    epoch2results = {}
    epoch_time = 0
    @property
    def nex_epoch(self):
        return int(self.epoch_time + self.tempo - c.time())

    def epoch(self):
        next_epoch = self.nex_epoch
        progress = c.tqdm(total=next_epoch, desc='Next Epoch')
        for _ in  range(next_epoch):
            progress.update(1)
            c.sleep(1)
        self.sync()
        executor = c.module('executor')(max_workers=self.max_workers, maxsize=self.batch_size)
        c.print(f'Epoch(network={self.network} epoch={self.epochs} n={self.n})', color='yellow')
        futures = []
        results = []
        progress = c.tqdm(total=self.n, desc='Evaluating Modules')
        # return self.modules
        for module in self.modules:
            c.print(f'EVAL --> {module})')
            if len(futures) < self.batch_size:
                futures.append(executor.submit(self.score_module, [module], timeout=self.timeout))
            else: 
                results.append(self.get_next_result(futures))
            progress.update(1)
        while len(futures) > 0:
            results.append(self.get_next_result(futures))
        results = [r for r in results if r.get('score', 0) > self.min_score]
        self.epochs += 1
        self.epoch_time = c.time()
        c.print(self.vote())
        return results
    
    def sync(self, update = False):

        network_path = self.path + '/state'
        max_age =  self.tempo or 60
        state = c.get(network_path, max_age=max_age, update=update)
        # RESOLVE THE VOTING NETWORKS
        network_module = c.module(self.network)() 
        modules = network_module.modules(subnet=self.subnet, max_age=max_age)
        params = network_module.params(subnet=self.subnet, max_age=max_age)
        self.tempo =  self.tempo or (params['tempo'] * network_module.block_time)//2
        self.params =  params
        state = {'time': c.time(), "params": params, 'modules': modules}
        if self.search != None:
            modules = [m for m in modules if self.search in m['name']]
        self.network_module = network_module
        self.n  = len(modules)  
        self.modules = modules
        self.network_info = {'n': self.n, 'network': self.network + '/' + str(self.subnet) if  self.network != 'local' else self.network, 'params': params}
        c.print(f'<Network({self.network_info})')
        return state

    module2last_update = {}

    def check_info(self, info:dict) -> bool:
        return bool(isinstance(info, dict) and all([k in info for k in  ['score', 'address', 'name', 'key']]))

    def score_module(self,  module:dict, **kwargs):
        module = c.copy(module)
        module['time'] = c.time() # the timestamp
        module_client = c.connect( module['address'], key=self.key)
        score = self.score(module_client, **kwargs)
        assert isinstance(score, (int, float)), f'Score is not a number {score}'
        module['score'] = score
        module['latency'] = c.time() - module['time']
        if module['score'] > self.min_score:
            module_path = self.path +'/'+ module['key']
            c.put_json(module_path, module)
        return module

    def votes(self, **kwargs):
        votes = {'keys': [], 'weights': []}
        for module in self.scoreboard().to_records():
            if module['score'] > 0:
                votes['keys'] += [module['key']]
                votes['weights'] += [module['score'].item()]
        return votes
    
    @property
    def votes_path(self):
        return self.path + f'/votes'

    def vote(self, update=False, submit_async=True, **kwargs):
        if not self.is_voting_network :
            return {'msg': f'NETWORK NOT VOTING NETWORK ({self.network}) out of ({self.voting_networks})', 'success': False,}
        if not hasattr(self, 'vote_time'):
            self.vote_time = 0
        vote_staleness = c.time() - self.vote_time
        if not update:
            if vote_staleness < self.tempo:
                return {'success': False, 'msg': f'Vote is too soon {vote_staleness}'}
        votes =self.votes() 
        return self.network_module.vote(modules=votes['keys'], 
                                        weights=votes['weights'], 
                                        key=self.key, 
                                        subnet=self.subnet)
    
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
            if isinstance(r, dict) and 'key' and  r.get('score', 0) > self.min_score  :
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
    
    def refresh_scoreboard(self):
        path = self.path
        c.rm(path)
        return {'success': True, 'msg': 'Leaderboard removed', 'path': path}
    
    @classmethod
    def run_epoch(cls, network='local', run_loop=False, **kwargs):
        return  cls(network=network, run_loop=run_loop, **kwargs).epoch()
    

    def get_next_result(self, futures):
        try:
            for future in c.as_completed(futures, timeout=self.timeout):
                futures.remove(future) 
                result = future.result()
                features = ['score', 'address', 'latency', 'name']
                if all([f in result for f in features]):
                    result = {k: result.get(k, 0) for k in features}
                    c.print(f'RESULT({result})')
                    return result
        except Exception as e:
            result = c.detailed_error(e)
        c.print(f'ERROR({result})')

        return result
    @staticmethod
    def test(  n=1, tag = 'vali_test_net',  miner='module', vali='vali',  path = '/tmp/commune/vali_test',network='local'):
        test_miners = [f'{miner}::{tag}_{i}' for i in range(n)]
        modules = test_miners
        search = tag
        for m in modules:
            c.serve(m)
        namespace = c.namespace(search=search)
        while len(namespace) < n:
            namespace = c.namespace(search=search)
        scoreboard = Vali.run_epoch(network=network, search=search, path=path)
        assert len(scoreboard) == n, f'Leaderboard not updated {scoreboard}'
        for miner in modules:
            c.print(c.kill(miner))
        return {'success': True, 'msg': 'subnet test passed'}
    
