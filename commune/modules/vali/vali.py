import torch
import traceback
import commune as c
import concurrent

class Vali(c.Module):
    
    network = 'subspace'
    worker_fn = 'worker'
    last_sync_time = 0
    last_evaluation_time = 0
    errors = 0
    count = 0
    requests = 0
    successes = 0
    n = 1

    def __init__(self,config:dict=None,**kwargs):
        self.init_vali(config=config, **kwargs)

    def init_vali(self,config=None, **kwargs):
        # initialize the validator
        config = self.set_config(config=config, kwargs=kwargs)

        # merge the config with the default config
        self.config = c.dict2munch({**Vali.config(), **config})

        # we want to make sure that the config is a munch
        self.start_time = c.time()
        self.sync_network()
        if self.config.run_loop:
            c.thread(self.run_loop)

    def run_info(self):
        info ={
            'lifetime': self.lifetime,
            'vote_staleness': self.vote_staleness,
            'errors': self.errors,
            'vote_interval': self.config.vote_interval,
            'epochs': self.epochs,
            'workers': self.workers()
        }
        return info
    def run_loop(self):
        
        if self.config.start:
            c.print(f'Vali config: {self.config}', color='cyan')
            if self.config.vote_tag == None:
                self.start_workers(num_workers=self.config.num_workers, 
                                   refresh=self.config.refresh, 
                                   mode=self.config.mode)
                steps = 0
                c.print(f'Vali loop started', color='cyan')
            while True:
                try:
                    steps += 1
                    c.print(f'Vali loop step {steps}', color='cyan')
                    run_info = self.run_info()
                    # sometimes the worker thread stalls, and you can just restart it
                    if 'subspace' in self.config.network:
                        if run_info['vote_staleness'] > self.config.vote_interval:
                            response = self.vote()
                            c.print(response)
                    c.print(run_info)
                    c.print('Sleeping... for {}', color='cyan')
                    c.sleep(self.config.run_loop_sleep)
                except Exception as e: 
                    c.print(e, color='red')


    
    def workers(self):
        return [f for f in c.pm2ls() if self.worker_name_prefix in f]
    
    def worker2logs(self, worker:str):
        workers = self.workers()
        worker2logs = {}
        for w in workers:
            worker2logs[w] = c.logs(w, lines=100)


    @property
    def worker_name_prefix(self):
        return f'{self.server_name}/{self.worker_fn}'

    def start_workers(self, 
                      num_workers:int=1, 
                      refresh=True, 
                      mode='process'):
        responses = []

        config= c.copy(self.config)
        config.start = False
        config.num_workers = 0
        config = c.munch2dict(config)
        if mode == 'process':

        # we don't want the workers to start more workers

            for i in range(num_workers):
                name = f'{self.worker_name_prefix}_{i}'
                if not refresh and c.pm2_exists(name):
                    c.print(f'Worker {name} already exists, skipping', color='yellow')
                    continue
                r = self.remote_fn(fn=self.worker_fn, 
                                name = name,
                                refresh=refresh,
                                kwargs={'config': config})
                c.print(f'Started worker {i} {r}', color='cyan')
                responses.append(r)

            return responses
        
        elif mode == 'thread':
            for i in range(num_workers):
                worker = c.thread(self.worker, kwargs=dict(config=config))
                c.print(f'Started worker {i} {worker}', color='cyan')
                responses.append(worker)
            return responses
        
    @classmethod
    def worker(cls, *args, **kwargs):
        kwargs['start'] = False
        self = cls(*args, **kwargs)
        c.new_event_loop(nest_asyncio=True)
        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')
        
        self.running = True
        last_print = 0

        self.executor  = c.module('executor.thread')(max_workers=self.config.num_threads)


        while self.running:
            results = []
            futures = []
            df_rows = []
            if self.last_sync_time + self.config.sync_interval < c.time():
                c.print(f'Syncing network {self.config.network}', color='cyan') 
                self.sync_network()
                
            module_addresses = c.shuffle(c.copy(self.module_addresses))
            batch_size = self.config.batch_size 
            # select a module
            results = []
            for  i, module_address in enumerate(module_addresses):
              
                if len(futures) < batch_size:
                    future = self.executor.submit(self.eval_module, args=[module_address], timeout=self.config.timeout)
                    futures.append(future)
                else:
                    
                    try:
                        for ready_future in c.as_completed(futures, timeout=self.config.timeout):
                            try:
                                result = ready_future.result()
                            except Exception as e:
                                result = {'success': False, 'error': c.detailed_error(e)}
                            futures.remove(ready_future)
                            results.append(result)
                            break
                    except Exception as e:
                        e = c.detailed_error(e)
                        c.print(f'Error {e}', color='red')
                        
    

                if c.time() - last_print > self.config.print_interval:
                    stats =  {
                        'lifetime': self.lifetime,
                        'pending': len(futures),
                        'sent': self.requests,
                        'errors': self.errors,
                        'successes': self.successes,
                            }
                    df_rows += [stats]
                    df = c.df(df_rows[-1:])
                    results = []
                    c.print(df)
                    last_print = c.time()


    def sync_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     update: bool = False):
        
        network = network or self.config.network
        search =  search or self.config.search
        netuid = netuid or self.config.netuid
        
        if 'subspace' in network:
            if '.' in network:
                """
                Assumes that the network is in the form of {{network}}.{{subnet/netuid}}
                """
                splits = network.split('.')
                assert len(splits) == 2, f'Network must be in the form of {{network}}.{{subnet/netuid}}, got {self.config.network}'
                network, netuid = splits
                netuid = int(netuid)
                network = network
                netuid = netuid
            else: 
                network = 'subspace'
                netuid = 0
                netuid = netuid
            self.subspace = c.module("subspace")(netuid=netuid)
            self.name2key = self.subspace.name2key(netuid=netuid)
        else:
            self.name2key = {}


        self.config.network = network
        self.config.netuid = netuid
        self.config.search = search

        self.namespace = c.namespace(search=search, 
                                    network=network, 
                                    netuid=netuid, 
                                    update=update)
        self.n  = len(self.namespace)    
        self.module_addresses = list(self.namespace.values())
        self.names = list(self.namespace.keys())
        self.address2name = {v: k for k, v in self.namespace.items()}    
        self.last_sync_time = c.time()
        
        r =  {
                'network': network, 
                'netuid': netuid, 
                'n': self.n, 
                'timestamp': self.last_sync_time,
                'msg': 'Synced network'
                }
        
        c.print(r)
        return r
        
        

    def score_module(self, module):
        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        return {'success': True, 'w': 1}

    

    def check_response(self, response:dict):
        """
        The following processes the response from the module
        """
        if type(response) in [int, float]:
            response = {'w': response}
        elif type(response) == bool:
            response = {'w': int(response)}
        else:
            assert isinstance(response, dict), f'Response must be a dict, got {type(response)}'
            assert 'w' in response, f'Response must have a w key, got {response.keys()}'

        return response
        

    def eval_module(self, module:str, network=None):
        return c.gather(self.async_eval_module(module=module, network=network))

    eval = eval_module

    def eval_batch(self, modules:list, **kwargs):
        return c.gather([self.async_eval_module(module=module, **kwargs) for module in modules])
    

    async def async_eval_module(self, module:str, network = None):
        """
        The following evaluates a module sver
        """
        # load the module stats (if it exists)
        if network != None:
            self.sync_network(network=network)

        namespace = self.namespace
        # RESOLVE THE MODULE ADDRESS
        module_address = None
        if module in namespace:
            module_name = module
            module_address = namespace[module]
        elif module in self.address2name:
            module_name = self.address2name[module]
            module_address = module
        else:
            module_name = module
        
        # load the module info and calculate the staleness of the module
        module_info = self.load_module_info( module_name, {})
        # if the module is stale, we can just return the module info
        seconds_since_called = c.time() - module_info.get('timestamp', 0)

        if seconds_since_called < self.config.max_staleness:
            return {'w': module_info.get('w', 0),
                     'module': module_name,
                       'address': module_address,
                         'timestamp': c.time(), 
                         'msg': f'Module is not stale, {int(seconds_since_called)} < {self.config.max_staleness}'}
        
        module_info['timestamp'] = c.time()

        try:
            self.requests += 1
            module = c.connect(module_address, key=self.key)
            if 'address' not in module_info:
                info = module.info()
                assert 'address' in info, f'Info must have a address key, got {info.keys()}'
                assert 'name' in info, f'Info must have a name key, got {info.keys()}'
                # we want to make sure that the module info has a timestamp
                module_info.update(info)
                
            
            module_name = module_info['name']
            # we want to make sure that the module info has a timestamp
            response = self.score_module(module)
            response = self.check_response(response)
            module_info.update(response)
            response['msg'] =  f'{c.emoji("checkmark")}{module_name} --> w:{response["w"]} {c.emoji("checkmark")} '
            self.successes += 1
        except Exception as e:
            e = c.detailed_error(e)
            response = { 'w': 0,'msg': f'{c.emoji("cross")} {module_name} --> {e} {c.emoji("cross")}'}  
            self.errors += 1  
        

        module_info['latency'] = c.time() - module_info['timestamp']
        
        # UPDATE W
        alpha = self.config.alpha # the weight of the new w
        w_old = module_info.get('w', 0) # the old weight
        w = response['w'] # the new weight
        module_info['w'] = w * alpha + w_old * (1 - alpha)

        path = self.resolve_storage_path(network=self.config.network, tag=self.tag) + f'/{module_name}'
        self.put_json(path, module_info)

        self.count += 1
        self.last_evaluation_time = c.time()

        response =  {'w': w, 
                     'module': module_name, 'address': module_address, 'timestamp': c.time(), 'path': path }
        return response

    @classmethod
    def resolve_storage_path(cls, network:str = 'subspace', tag:str=None):
        tag = tag or 'base'
        return f'{tag}.{network}'
        
    def refresh_stats(self, network='subspace', tag=None):
        path = self.resolve_storage_path(network=network, tag=tag)
        return self.rm(path)
    
    def resolve_tag(self, tag:str=None):
        return tag or self.config.vote_tag or self.tag
    
    def vote_stats(self, votes = None, tag=None):
        votes = votes or self.load_votes()
        tag = self.resolve_tag(tag)
        info = {
            'num_uids': len(votes['uids']),
            'avg_weight': c.mean(votes['weights']),
            'stdev_weight': c.stdev(votes['weights']),
            'timestamp': votes['timestamp'],
            'lag': c.time() - votes['timestamp'],
            'tag': tag,
        }
        return info
    
    def votes(self, tag=None, network = None):
        network = network or self.config.network
        tag = tag or self.config.vote_tag or self.tag
        c.print(f'Calculating votes for {network} {tag}', color='cyan')

        # get the list of modules that was validated
        module_infos = self.module_infos(network=network, keys=['name', 'w', 'ss58_address'], tag=tag)
        votes = {
            'keys' : [],            # get all names where w > 0
            'weights' : [],  # get all weights where w > 0
            'uids': [],
            'timestamp' : c.time() 
        }

        key2uid = self.subspace.key2uid()
        for info in module_infos:
            if 'ss58_address' in info and info['w'] >= 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid[info['ss58_address']]]

        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes

    def load_votes(self, tag=None) -> dict:
        tag = self.resolve_tag(tag)
        default={'uids': [], 'weights': [], 'timestamp': 0, 'block': 0}
        votes = self.get(f'votes/{self.config.network}/{self.tag}', default=default)
        return votes

    def save_votes(self, votes:dict):
        assert isinstance(votes, dict), f'Weights must be a dict, got {type(votes)}'
        assert 'uids' in votes, f'Weights must have a uids key, got {votes.keys()}'
        assert 'weights' in votes, f'Weights must have a weights key, got {votes.keys()}'
        assert 'timestamp' in votes, f'Weights must have a timestamp key, got {votes.keys()}'
        storage_path = self.resolve_storage_path(network=self.config.network, tag=self.tag)
        self.put(f'votes/{self.config.network}/{self.tag}', votes)

    @classmethod
    def tags(cls, network=network, mode='stats'):
        return list([p.split('/')[-1].split('.')[0] for p in cls.ls()])

    @classmethod
    def paths(cls, network=network, mode='stats'):
        return list(cls.tag2path(network=network, mode=mode).values())

    @classmethod
    def tag2path(cls, network:str=network, mode='stats'):
        return {f.split('/')[-1].split('.')[0]: f for f in cls.ls(f'{mode}/{network}')}

    @classmethod
    def sand(cls):
        for path in cls.ls('votes'):
            if '/main.' in path:
                new_path = c.copy(path)
                new_path = new_path.replace('/main.', '/main/')
                c.mv(path, new_path)

    @classmethod
    def module_paths(cls, network:str=network, tag:str=None):
        path = cls.resolve_storage_path(network=network, tag=tag)
        paths = cls.ls(path)
        return paths


    def vote(self, tag=None, votes=None, cache_exceptions=True):
        if cache_exceptions:
            try:
                response =  self.vote(tag=tag, votes=votes, cache_exceptions=False)
            except Exception as e:
                response = c.detailed_error(e)
            return response 

        c.print(f'Voting {self.config.network} {self.config.netuid}', color='cyan')

        votes = votes or self.votes(tag=tag) 
        if tag != None:
            key = self.resolve_server_name(tag=tag)
            key = c.get_key(key)
        else:
            key = self.key

        if len(votes['uids']) < self.config.min_num_weights:
            response = {'success': False, 'msg': 'The votes are too low', 'votes': len(votes['uids']), 'min_num_weights': self.config.min_num_weights}
            return response

        r = c.vote(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                        weights=votes['weights'], 
                        key=self.key, 
                        network=self.config.network, 
                        netuid=self.config.netuid)

        self.save_votes(votes)

        return {'success': True, 
                'message': 'Voted', 
                'num_uids': len(votes['uids']),
                'avg_weight': c.mean(votes['weights']),
                'stdev_weight': c.stdev(votes['weights']),
                'r': r}

    @classmethod
    def module_names(cls, network:str=network, tag:str=None):
        paths = cls.module_paths(network=network, tag=tag)
        modules = [p.split('/')[-1].replace('.json', '') for p in paths]
        return modules

    @classmethod
    def num_module_infos(cls, tag=None, network=network, **kwargs):
        return len(cls.module_infos(network=network,tag=tag, **kwargs))
    


    @classmethod
    def leaderboard(cls, *args, **kwargs): 
        df =  c.df(cls.module_infos(*args, **kwargs))
        df.sort_values(by=['w', 'staleness'], ascending=False, inplace=True)
        return df
        
    @classmethod
    def module_infos(cls,
                    tag=None,
                    network:str='subspace', 
                    batch_size:int=100 , # batch size for 
                    max_staleness:int= 3600,
                    update = True,
                    keys = ['name', 'w', 'staleness', 'timestamp', 'address', 'ss58_address'],
                    path = 'cache/module_infos'
                    ):
        
        if not update:
            path = cls.resolve_storage_path(network=network, tag=tag) + f'/{path}'
            module_infos = cls.get(path, [])
            if len(module_infos) > 0:
                return module_infos

        paths = cls.module_paths(network=network, tag=tag)   
        jobs = [c.async_get_json(p) for p in paths]
        module_infos = []
        # chunk the jobs into batches
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.gather(jobs_batch)
            # last_interaction = [r['history'][-1][] for r in results if r != None and len(r['history']) > 0]
            for s in results:

                if not isinstance(s, dict):
                    continue
                s['staleness'] = c.timestamp() - s.get('timestamp', 0)
                if keys != None and not all([k in s for k in keys]):
                    # the module info is missing some keys
                    continue
                if s == None:
                    continue
                if s['staleness'] > max_staleness:
                    continue
                if s['w'] <= 0:
                    continue
                if keys  != None:
                    s = {k: s.get(k,None) for k in keys}
                module_infos += [s]

        cls.put(path, module_infos)

        return module_infos
    

    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_info(self, k:str,default=None):
        default = default if default != None else {}
        path = self.resolve_storage_path(network=self.config.network, tag=self.tag) + f'/{k}'
        return self.get_json(path, default=default)


    def get_history(self, k:str, default=None):
        module_infos = self.load_module_info(k, default=default)
        return module_infos.get('history', [])
    


    @property
    def last_vote_time(self):
        votes = self.load_votes()
        return votes.get('timestamp', 0)
    
    @property
    def vote_staleness(self) -> int:
        return int(c.time() - self.last_vote_time)

    @property
    def epochs(self):
        return self.count // (self.n + 1)
    
    def stop(self):
        self.running = False
    
    @classmethod
    def check_loop_running(cls):
        return c.pm2_exists(cls.check_loop_name)

    @classmethod
    def ensure_check_loop(self):
        if self.check_loop_running() == False:
            self.check_loop(remote=True)

    @property
    def lifetime(self):
        return c.time() - self.start_time

    def modules_per_second(self):
        return self.count / self.lifetime

    @classmethod
    def test(cls, **kwargs):
        kwargs['num_workers'] = 0
        kwargs['vote'] = False
        kwargs['verbose'] = True
        self = cls(**kwargxs )
        return self.run()

    @classmethod
    def dashboard(cls):
        import streamlit as st
        # disable the run_loop to avoid the background  thread from running
        self = cls(start=False)
        c.load_style()
        module_path = self.path()
        network = 'local'
        c.new_event_loop()
        
        st.title(module_path)

        servers = c.servers(search='vali')
        server = st.selectbox('Select Vali', servers)
        state_path = f'dashboard/{server}'
        module = c.module(server)
        state = module.get(state_path, {})
        server = c.connect(server)
        if len(state) == 0 :
            state = {
                'run_info': server.run_info(),
                'module_infos': server.module_infos()
            }

            self.put(state_path, state)



        my_modules = c.my_modules()
        
        # st.write(my_modules)


        run_info = state['run_info']
        module_infos = state['module_infos']
        df = []
        

        default_columns = ['name', 'staleness', 'w', 'timestamp', 'address']
        for row in module_infos:
            if isinstance(row, dict):
                columns = list(row.keys())
                break
        selected_columns = default_columns

        search = st.text_input('Search')
        
        for row in module_infos:
            row['name'] = row.get('name', '')
            if search != '' and search not in row['name']:
                continue
            
            row = {k: row.get(k, None) for k in selected_columns}
            df += [row]
        df = c.df(df)
        if len(df) == 0:
            st.write('No modules found')
            return
        df.sort_values(by=['w', 'staleness'], ascending=False, inplace=True)

        st.write(df)

    def __del__(self):
        self.stop()
        c.print(f'Vali {self.config.network} {self.config.netuid} stopped', color='cyan')
        workers = self.workers()
        futures = []
        for w in workers:
            c.print(f'Stopping worker {w}', color='cyan')
            futures += [c.submit(c.kill, args=[w])]
        return c.wait(futures, timeout=10)
        


        

        # columns = list(df.columns)
        # columns.pop(columns.index('stake_from'))
        # with st.expander('Columns'):
        #     default_columns = ['name', 'stake', 'dividends', 'last_update', 'delegation_fee']
        #     columns = st.multiselect('Select columns',columns ,default_columns )
        # df = df[columns]

        # namespace = c.namespace(search=module_path)

        # st.write(df)
        # c.plot_dashboard(df)
        
Vali.run(__name__)
