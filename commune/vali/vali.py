
import commune as c
class Vali(c.Module):
    
    last_sync_time = 0
    last_vote_time = 0
    last_sent = 0
    last_success = 0
    errors = 0
    count = 0
    requests = 0
    successes = 0
    epochs = 0
    n = 1
    whitelist = []

    def __init__(self,config:dict=None,**kwargs):
        self.init_vali(config=config, **kwargs)

    def init_vali(self,config=None, **kwargs):
        # initialize the validator
        config = self.set_config(config=config, kwargs=kwargs)
        # merge the config with the default config
        self.config = c.dict2munch({**Vali.config(), **config})
        # we want to make sure that the config is a munch
        self.sync()
        if self.config.workers > 0:    
            c.thread(self.start)


    def run_info(self):
        info ={
            'vote_block_staleness': self.block_staleness,
            'errors': self.errors,
            'vote_interval': self.config.vote_interval,
            'epochs': self.epochs,
            'workers': self.workers(),
        }
        return info
    

    
    def workers(self):
        if self.config.mode == 'server':
            return c.servers(search=self.server_name)
        elif self.config.mode == 'thread':
            return c.threads(search='worker')
        else:
            return []
        
    
    def worker2logs(self):
        workers = self.workers()
        worker2logs = {}
        for w in workers:
            worker2logs[w] = c.logs(w, lines=100)


    @property
    def worker_name_prefix(self):
        return f'{self.server_name}'



    def add_worker(self):
        num_workers = len(self.workers())
        id = num_workers
        return self.start_worker(id=id)
    

    def restart_worker(self, id = 0):
        if self.config.mode == 'thread':
            return self.start_worker(id=id)
        



    def start_worker(self, id = 0, **kwargs):
        config = self.config
        config = c.munch2dict(config) # we want to convert the config to a dict
        worker_name = self.worker_name(id)
        kwargs={'config': config, 'id': id, 'workers':0, **kwargs}
        if self.config.mode == 'thread':
            worker = c.thread(self.worker, kwargs=kwargs, name=worker_name)
        elif self.config.mode == 'process':
            worker = c.process(self.worker, kwargs=kwargs, name=worker_name)
        elif self.config.mode == 'server':
            worker = self.serve(kwargs=kwargs, 
                                 key=self.key.path, 
                                 server_name = worker_name or self.server_name)
        else:
            raise Exception(f'Invalid mode {self.config.mode}')
        c.print(f'Started worker {worker}', color='cyan')

        return {'success': True, 'msg': f'Started worker {worker}', 'worker': worker}

    def worker_name(self, id = 0):
        return f'{self.config.worker_fn_name}::{id}'


    @classmethod        
    def worker(cls, config = None, id = 0, **kwargs):
        color = c.random_color()

        self = cls(config=config, **kwargs)
        worker_name = self.worker_name(id)

        batch_size = self.config.batch_size 
        self.running = True
        last_print = 0
        executor = c.module('executor.thread')(max_workers=self.config.threads_per_worker)
        futures = []

        while self.running:
            
            if c.time() - self.last_sync_time > self.config.sync_interval:
                self.sync()

            module_addresses = c.shuffle(list(self.namespace.values()))
            batch_size = min(batch_size, len(module_addresses))
            
            # select a module
            for  i, module_address in enumerate(module_addresses):
                

                # if the futures are less than the batch, we can submit a new future
                if len(futures) < batch_size:
                    self.last_sent = c.time()
                    future = executor.submit(self.eval_module, args=[module_address], timeout=self.config.timeout)
                    futures.append(future)
                else:
                    try:
                        for ready_future in c.as_completed(futures, timeout=self.config.timeout):
                            result = ready_future.result()

                            if c.is_error(result):
                                self.errors += 1
                            else:
                                self.successes += 1
                                self.last_success = c.time()
                            c.print(result, verbose=self.config.verbose or self.config.debug)

                            futures.remove(ready_future)
                            break
                    except Exception as e:
                        self.errors += 1
                        result = c.detailed_error(e)
                        c.print(result, verbose=self.config.verbose or self.config.debug)

                        continue

                    
                    
                if c.time() - last_print > self.config.print_interval:
                    stats =  {
                        'successes': self.successes,
                        'sent': self.requests,
                        'errors': self.errors,
                        'network': self.network,
                        'pending': len(futures),
                        'vote_staleness': self.block_staleness,
                        'last_success': c.round(c.time() - self.last_success,2),
                        'last_sent': c.round(c.time() - self.last_sent,2),
                        'worker_name': worker_name,
                            }
                    self.put(f'clone_stats/{worker_name}', stats)
                    c.print(stats, color=color)
                    last_print = c.time()
                


    def clone_stats(self):
        workers = self.workers()
        stats = {}
        for w in workers:
            stats[w] = self.get(f'clone_stats/{w}', default={})
        return stats

    def set_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     fn : str = None,
                     update: bool = False):
        
        network = network or self.config.network
        search =  search or self.config.search
        netuid = netuid or self.config.netuid
        fn = fn or self.config.fn
        
        # this is only for 
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
        else:
            self.name2key = {}

        # name2address / namespace

        self.namespace = c.module('namespace').namespace(search=search, 
                                    network=network, 
                                    netuid=netuid, 
                                    update=update)
        self.n  = len(self.namespace)    
        self.address2name = {v: k for k, v in self.namespace.items()}    
        self.last_sync_time = c.time()

        self.network = self.config.network = network
        self.netuid = self.config.netuid = netuid
        self.search = self.config.search = search
        self.fn = self.config.fn = fn or self.config.fn
        
        r = {
                'search': search,
                'network': network, 
                'netuid': netuid, 
                'n': self.n, 
                'msg': 'Synced network'
                }
        c.print(r)
        return r
    sync = set_network
        
    def score_module(self, module: 'c.Module'):
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
        
    def get_module_info(self, module):
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

        info = self.load_module_info( module_name, {})
        info['address'] = module_address
        info['name'] = module_name
        info['schema'] = info.get('schema', None)

        return info
    

    def eval_module(self, module:str):
        """
        The following evaluates a module sver
        """
        # load the module stats (if it exists)

        
        # load the module info and calculate the staleness of the module
        # if the module is stale, we can just return the module info

        info = self.get_module_info(module)

        module = c.connect(info['address'], key=self.key)

        self.requests += 1

        seconds_since_called = c.time() - info.get('timestamp', 0)
        if seconds_since_called < self.config.max_staleness:
            return {'w': info.get('w', 0),
                    'module': info['name'],
                    'address': info['address'],
                        'timestamp': c.time(), 
                        'msg': f'Module is not stale, {int(seconds_since_called)} < {self.config.max_staleness}'}
        else:
            module_info = module.info(timeout=self.config.timeout)
            assert 'address' in info and 'name' in info
            # we want to make sure that the module info has a timestamp
            info.update(module_info)
            info['timestamp'] = c.time()

        try:
            # we want to make sure that the module info has a timestamp
            response = self.score_module(module)
            response = self.check_response(response)
            info.update(response)            
            self.successes += 1
        except Exception as e:
            e = c.detailed_error(e)
            response = { 'w': 0,'msg': f'{c.emoji("cross")} {info["name"]} {c.emoji("cross")}'}  
        
        info['latency'] = c.time() - info['timestamp']
        info['w'] = response['w']  * self.config.alpha + info['w'] * (1 - self.config.alpha)
        path = f'{self.storage_path()}/{info["name"]}'
        self.put_json(path, info)
        return {'w': info['w'], 'module': info['name'], 'address': info['address'], 'latency': info['latency']}
        

    def storage_path(self):
        network = self.network
        if 'subspace' in network:
            network_str = f'{network}.{self.netuid}'
        else:
            network_str = network
            
        path =  f'{network_str}'

        return path
        
    
    def resolve_tag(self, tag:str=None):
        return tag or self.config.vote_tag or self.tag
    
    def vote_stats(self, votes = None):
        votes = votes or self.votes()
        info = {
            'num_uids': len(votes['uids']),
            'avg_weight': c.mean(votes['weights']),
            'stdev_weight': c.stdev(votes['weights']),
            'timestamp': votes['timestamp'],
            'lag': c.time() - votes['timestamp'],
        }
        return info
    
    def votes(self):
        network = self.network
        module_infos = self.module_infos(network=network, keys=['name', 'w', 'ss58_address'])
        votes = {'keys' : [],'weights' : [],'uids': [], 'timestamp' : c.time()  }
        key2uid = self.subspace.key2uid()
        for info in module_infos:
            ## valid modules have a weight greater than 0 and a valid ss58_address
            if 'ss58_address' in info and info['w'] >= 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid[info['ss58_address']]]

        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes
    
    @property
    def votes_path(self):
        return self.storage_path() + f'/votes'

    def load_votes(self) -> dict:
        return self.get(self.votes_path, default={'uids': [], 'weights': [], 'timestamp': 0, 'block': 0})

    def save_votes(self, votes:dict):
        assert isinstance(votes, dict), f'Weights must be a dict, got {type(votes)}'
        assert 'uids' in votes, f'Weights must have a uids key, got {votes.keys()}'
        assert 'weights' in votes, f'Weights must have a weights key, got {votes.keys()}'
        assert 'timestamp' in votes, f'Weights must have a timestamp key, got {votes.keys()}'
        self.put(self.votes_path, votes)





    def vote(self, async_vote:bool=False, save:bool = True, **kwargs):
        


        if async_vote:
            return c.submit(self.vote, **kwargs)

        votes =self.votes() 

        if len(votes['uids']) < self.config.min_num_weights:
            return {'success': False, 'msg': 'The votes are too low', 'votes': len(votes['uids']), 'min_num_weights': self.config.min_num_weights}

        r = c.vote(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                        weights=votes['weights'], 
                        key=self.key, 
                        network=self.config.network, 
                        netuid=self.config.netuid)
        
        if save:
            self.save_votes(votes)

        self.last_vote_time = c.time()
        
        return {'success': True, 
                'message': 'Voted', 
                'num_uids': len(votes['uids']),
                'timestamp': self.last_vote_time,
                'avg_weight': c.mean(votes['weights']),
                'stdev_weight': c.stdev(votes['weights']),
                'r': r}
    
    @classmethod
    def num_module_infos(cls, tag=None, network='subspace', **kwargs):
        return len(cls.module_infos(network=network,tag=tag, **kwargs))

    @classmethod
    def leaderboard(cls, *args, **kwargs): 
        df =  c.df(cls.module_infos(*args, **kwargs))
        df.sort_values(by=['w', 'staleness'], ascending=False, inplace=True)
        return df
    
    @property
    def module_paths(self):
        paths = self.ls(self.storage_path())
        paths = list(filter(lambda x: x.endswith('.json'), paths))
        return paths
    

    @property
    def module_info(self):
        return self.subspace.get_module(self.key.ss58_address, netuid=self.netuid)
    
    def module_infos(self,
                    batch_size:int=100 , # batch size for 
                    timeout:int=10,
                    keys = ['name', 'w', 'staleness', 'timestamp', 'address', 'ss58_address'],
                    path = 'cache/module_infos',
                    update = True,
                    **kwargs
                    ):
        
        if not update:
            modules_info = self.get(path, default=[])
            if len(modules_info) > 0:
                return modules_info
            
        paths = self.module_paths
        jobs = [c.async_get_json(p) for p in paths]
        module_infos = []
        # chunk the jobs into batches
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.wait(jobs_batch, timeout=timeout)
            # last_interaction = [r['history'][-1][] for r in results if r != None and len(r['history']) > 0]
            for s in results:
                if isinstance(s, dict) and 'ss58_address' in s:
                    s['staleness'] = c.time() - s.get('timestamp', 0)
                    module_infos += [{k: s.get(k, None) for k in keys}]

        if update:
            self.put(path, module_infos)       
        return module_infos

    def load_module_info(self, k:str,default=None):
        default = default if default != None else {}
        path = self.storage_path() + f'/{k}'
        return self.get_json(path, default=default)
    
    def save_module_info(self, k:str, v:dict):
        path = self.storage_path() + f'/{k}'
        self.put_json(path, v)

    def get_history(self, k:str, default=None):
        module_infos = self.load_module_info(k, default=default)
        return module_infos.get('history', [])
    
    
    @property
    def vote_staleness(self) -> int:
        return int(c.time() - self.last_vote_time)
    
    def stop(self):
        self.running = False

    def __del__(self):
        self.stop()
        c.print(f'Vali {self.config.network} {self.config.netuid} stopped', color='cyan')
        workers = self.workers()
        futures = []
        for w in workers:
            if self.config.mode == 'thread': 
                c.print(f'Stopping worker {w}', color='cyan')
                futures += [c.submit(c.kill, args=[w])]
            elif self.config.mode == 'server':
                c.print(f'Stopping server {w}', color='cyan')
                futures += [c.submit(c.kill, args=[w])]
        return c.wait(futures, timeout=10)
        
    @classmethod
    def dashboard(cls):
        import streamlit as st
        # disable the run_loop to avoid the background  thread from running
        self = cls(start=False)
        c.load_style()
        module_path = self.path()
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
                'run_info': server.run_info,
                'module_infos': server.module_infos(update=True)
            }

            self.put(state_path, state)

        module_infos = state['module_infos']
        df = []
        selected_columns = ['name', 'address', 'w', 'staleness']

        selected_columns = st.multiselect('Select columns', selected_columns, selected_columns)
        search = st.text_input('Search')

        for row in module_infos:
            if search != '' and search not in row['name']:
                continue
            row = {k: row.get(k, None) for k in selected_columns}
            df += [row]
        df = c.df(df)
        if len(df) == 0:
            st.write('No modules found')
        else:
            default_columns = ['w', 'staleness']
            sorted_columns = [c for c in default_columns if c in df.columns]
            df.sort_values(by=sorted_columns, ascending=False, inplace=True)
        st.write(df)

    def random_module(self):
        return c.choice(list(self.namespace.keys()))

    @classmethod
    def test_eval_module(cls, network='local', verbose=False, timeout=1, workers=2, start=False,  **kwargs):
        self = cls(network=network, workers=workers, verbose=verbose, timeout=timeout, start=start,  **kwargs)
        return self.eval_module(self.random_module())


    @classmethod
    def test_network(cls, network='subspace', search='vali'):
        server_name = 'vali::test'
        self = cls(search=search, network=network, start=False, workers=0)
        if len(self.namespace) > 0:
            for module_name in self.namespace:
                assert search in module_name
        c.kill(server_name)
        return {'success': True, 'msg': f'Found {len(self.namespace)} modules in {network} {search}'}


    def start(self):
        # start the workers
        for i in range(self.config.workers):
            self.start_worker(i)
        c.thread(self.vote_loop)

    @property
    def should_vote(self) -> bool:
        is_voting_network = 'subspace' in self.config.network or 'bittensor' in self.config.network
        is_stale = self.vote_staleness > self.config.vote_interval
        should_vote = is_voting_network and is_stale
        return should_vote
    

    @property
    def block_staleness(self):
        return self.subspace.block - self.module_info['last_update']
    

    
    def vote_loop(self):

        while True:
            if self.should_vote:
                try:
                    r = self.vote()
                except Exception as e:
                    r = c.detailed_error(e)
                    c.print(c.detailed_error(e))
            run_info = self.run_info()
            df = c.df([run_info])
            c.print(color='cyan')
            c.sleep(self.config.sleep_interval)


        

        
Vali.run(__name__)
