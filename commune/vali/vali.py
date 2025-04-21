
import commune as c
import os
import pandas as pd
from typing import *
import inspect
class Vali:
    endpoints = ['score', 'scoreboard']
    def __init__(self,

                    network= 'local', # for local chain:test or test # for testnet chain:main or main # for mainnet
                    search : Optional[str] =  None, # (OPTIONAL) the search string for the network 
                    batch_size : int = 128, # the batch size of the most parallel tasks
                    task : str= 'task', # score function
                    key : str = None, # the key for the module
                    tempo : int = 60, # the time between epochs
                    timeout : int = 3, # timeout per evaluation of the module
                    update : bool =True, # update during the first epoch
                    run_loop : bool = True, # This is the key that we need to change to false
                    verbose: bool = True, # print verbose output
                    path : str= None, # the storage path for the module eval, if not null then the module eval is stored in this directory
                 **kwargs): 

        self.epoch_time = 0
        self.vote_time = 0 # the time of the last vote (for voting networks)
        self.epochs = 0 # the number of epochs
        self.subnet = None    
        self.timeout = timeout
        self.batch_size = batch_size
        self.verbose = verbose
        self.key = c.get_key(key)
        self.set_task(task)
        self.set_network(network=network, tempo=tempo,  search=search,  path=path, update=update)
        if run_loop:
            c.thread(self.run_loop) if run_loop else ''

    def set_network(self, 
                    network:Optional[str] = None, 
                    tempo:int= 10, 
                    search:str=None, 
                    path:str=None, 
                    update = False):
        if not hasattr(self, 'network'):
            self.network = 'local'
        self.network = network or self.network
        self.tempo = tempo
        self.storage_path = self.get_path(self.network + '/' + self.network)
        self.search = search
        self.net = c.module(self.network)() 
        self.sync(update=update)

    def sync(self, update=False):
        self.params = self.net.params(subnet=self.subnet, max_age=self.tempo, update=update)
        self.modules = self.net.modules(subnet=self.subnet, max_age=self.tempo, update=update)        
        # create some extra helper mappings
        self.key2module = {m['key']: m for m in self.modules if 'key' in m}
        self.name2module = {m['name']: m for m in self.modules if 'name' in m}
        self.url2module = {m['url']: m for m in self.modules if 'url' in m}
        if self.search:
            self.modules = [m for m in self.modules if any(str(self.search) in str(v) for v in m.values())]
        return self.params
    
    def set_task(self, task: Union[str, 'callable', int]):
        if isinstance(task, str):
            task = c.module(task)()
        assert hasattr(task, 'forward'), f'Task {task} does not have a forward method'
        self.task = task
        task_path = task.__module__ + '.' + task.__class__.__name__
        task_code = c.code_map(task_path)
        task_hash = c.hash(task_code)
        self.task.info  = {
            'name': task.__class__.__name__.lower(),
            'schema': c.schema(task_path),
            'code': task_code,
            'cid': task_hash, # the content id of the code of the task
        }

    def get_path(self, path):
        return os.path.expanduser(f'~/.commune/vali/{path}')

    def next_epoch_time(self):
        return self.epoch_time + self.tempo

    def seconds_until_epoch(self):
        return int(self.next_epoch_time() - c.time())
    
    def run_loop(self, step_time=2):
        while True:
            # wait until the next epoch)
            seconds_until_epoch = self.seconds_until_epoch()
            if seconds_until_epoch > 0:
                progress = c.tqdm(total=seconds_until_epoch, desc='Time Until Next Progress')
                for i in range(seconds_until_epoch):
                    progress.update(step_time)
                    c.sleep(step_time)
            try:
                c.df(self.epoch())
            except Exception as e:
                c.print('XXXXXXXXXX EPOCH ERROR ----> XXXXXXXXXX ',c.detailed_error(e), color='red')



    def get_module(self, module:Union[str, dict]):
        if isinstance(module, str):
            if module in self.key2module:
                module = self.key2module[module]
            elif module in self.name2module:
                module = self.name2module[module]
            elif module in self.url2module:
                module = self.url2module[module]
            else:
                raise ValueError(f'Module not found {module}')
        return module
    def forward(self,  module:Union[str, dict], **params):
        module = self.get_module(module)
        module['time'] = c.time()
        client = c.client(module['url'], key=self.key)
        c.print(f'Sample(task={self.task.info["name"]} module={module["name"]} url={module["url"]})')
        result = self.task.forward(client, **params)
        # prepare the module for the result
        assert 'score' in result, f'Module {module["name"]} does not have a score {result}'
        data = {**module, **result}
        data['params'] = params
        data['result'] = result
        data['time'] = c.time()
        data['duration'] = c.time() - module['time']
        data['vali'] = self.key.key_address
        data['task'] = self.task.info["name"]
        data['path'] = self.get_module_path(data['key'])
        data['proof'] = c.sign(c.hash(data), key=self.key, mode='dict')
        self.verify_proof(data) # verify the proof
        c.put_json(data['path'], data)
        return data

    def get_module_path(self, module:str):
        return self.storage_path + '/' + module + '.json'

    def module_results(self, module: Union[str, dict]):
        path = self.get_module_path(module)
        return c.get_json(path)

    def verify_proof(self, module:dict):
        module = c.copy(module)
        proof = module.pop('proof', None)
        assert c.verify(proof), f'Invalid Proof {proof}'

    def epoch(self, features=['score', 'key', 'duration', 'name'], **kwargs):
        self.sync()
        n = len(self.modules)
        batches = [self.modules[i:i+self.batch_size] for i in range(0, n, self.batch_size)]
        num_batches = len(batches)
        epoch_info = {
            'epochs' : self.epochs,
            'task': self.task.info['name'],
            'key': self.key.key_address,
            'batch_size': self.batch_size,
        }
        results = []
        for i, batch in enumerate(batches):
            futures = []
            future2module = {}
            for m in batch:
                print(f'Batch {i}/{num_batches} {m["name"]} {m["url"]}')
                future = c.submit(self.forward, [m], timeout=self.timeout)
                future2module[future] = m
                
            for future in c.as_completed(future2module):
                try:
                    m = future2module[future]
                    result = future.result()
                    if isinstance(result, dict) and 'score' in result:
                        results.append(result)
                    else: 
                        c.print(f'Error({m["name"]}, result={result})')
                except Exception as e:
                    print(f'Error in batch {i} {c.detailed_error(e)}')

        self.epochs += 1
        self.epoch_time = c.time()
        self.vote(results)
        if len(results) > 0:
            return c.df(results)[features].sort_values(by='score', ascending=False)
        else:
            return c.df([{'success': False, 'msg': 'No results to vote on', 'epoch_info': epoch_info}])

    @property
    def vote_staleness(self):
        return c.time() - self.vote_time

    def vote(self, results):
        if not bool(hasattr(self.net, 'vote')) :
            return {'success': False, 'msg': f'NOT VOTING NETWORK({self.network})'}
        if self.vote_staleness < self.tempo:
            return {'success': False, 'msg': f'Vote is too soon {self.vote_staleness}'}
        if len(results) == 0:
            return {'success': False, 'msg': 'No results to vote on'}
        # get the top modules
        assert all('score' in r for r in results), f'No score in results {results}'
        assert all('key' in r for r in results), f'No key in results {results}'
        return self.net.vote(
                    modules=[m['key'] for m in modules], 
                    weights=[m['score'] for m in modules],  
                    key=self.key, 
                    subnet=self.subnet
                    )
    
    def results(self,
                    keys = ['name', 'score', 'duration',  'url', 'key', 'time', 'age'],
                    ascending = True,
                    by = 'score',
                    to_dict = False,
                    page = None,
                    max_age = 10000,
                    update= False,
                    **kwargs
                    ) -> Union[pd.DataFrame, List[dict]]:
        page_size = 1000
        df = []
        # chunk the jobs into batches
        for path in c.ls(self.storage_path):
            r = c.get(path, {},  max_age=max_age, update=update)
            if isinstance(r, dict) and 'key' and  r.get('score', 0) > 0  :
                df += [{k: r.get(k, None) for k in keys}]
            else :
                c.print(f'REMOVING({path})', color='red')
                c.rm(path)
        df = c.df(df) 
        if len(df) > 0:
            if isinstance(by, str):
                by = [by]
            df = df.sort_values(by=by, ascending=ascending)
        if len(df) > page_size:
            pages = len(df)//page_size
            page = page or 0
            df = df[page*page_size:(page+1)*page_size]
        df['age'] = c.time() - df['time']
        if to_dict:
            return df.to_dict(orient='records')
        return df

    @classmethod
    def run_epoch(cls, network='local', **kwargs):
        kwargs['run_loop'] = False
        return  cls(network=network,**kwargs).epoch()
    
    def refresh_results(self):
        path = self.storage_path
        c.rm(path)
        return {'success': True, 'msg': 'Leaderboard removed', 'path': path}

    def tasks(self):
        return c.modules(search='task.')

    @classmethod
    def test(cls ,  n=2, 
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
            scoreboard = []
            while len(scoreboard) < n:
                c.sleep(1)
                scoreboard = vali.epoch()
                trials -= 1
                assert trials > 0, f'Trials exhausted {trials}'
            for miner in modules:
                c.print(c.kill(miner))
            assert c.server_exists(miner) == False, f'Miner still exists {miner}'
            return {'success': True, 'msg': 'subnet test passed'}