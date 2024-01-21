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
            self.start_workers(num_workers=self.config.num_workers, refresh=self.config.refresh)
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
                            self.vote()
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

    def start_workers(self, num_workers:int=1, refresh=True):
        responses = []

        config= c.copy(self.config)
        config.start = False
        config.num_workers = 0
        config = c.munch2dict(config)

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
    

        
    @classmethod
    def worker(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        c.new_event_loop(nest_asyncio=True)
        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')
        
        self.running = True
        last_print = 0

        self.executor  = c.module('executor.thread')(max_workers=self.config.threads_per_worker)

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
                    
                    for ready_future in c.as_completed(futures, timeout=self.config.timeout):
                        try:
                            result = ready_future.result()
                        except Exception as e:
                            continue

                        results.append(result)
                        futures.remove(ready_future)
                        break
    

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
                    import sys
                    sys.stdout.flush()
                    c.print(df)
                    last_print = c.time()
                    # c.print(f'STATS  --> {stats}\n', color='white')



    def sync_network(self, 
                     network:str=None, 
                     search:str=None,  
                     netuid:int=None, 
                     update: bool = False):
        if network == None:
            network = self.config.network
        if search == None:
            search = self.config.search
        if netuid == None:
            netuid = self.config.netuid
        


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

        self.namespace = c.namespace(search=self.config.search, 
                                    network=self.config.network, 
                                    netuid=self.config.netuid, 
                                    update=update)
        self.n  = len(self.namespace)    
        self.module_addresses = list(self.namespace.values())
        self.names = list(self.namespace.keys())
        self.address2name = {v: k for k, v in self.namespace.items()}    
        self.last_sync_time = c.time()
        c.print(f'Synced network {self.config.network} netuid {self.config.netuid} n {self.n}', color='cyan')
        return {
                'network': self.config.network, 
                'netuid': self.config.netuid, 
                'n': self.n, 
                'timestamp': self.last_sync_time
                }
        
        

    def score_module(self, module):
        '''
        params:
            module: module client
            kwargs : the key word arguments
        
        '''
        # info = module.info()
        # assert 'name' in info, f'Info must have a name key, got {info.keys()}'
        w = 1

        # assert 'address' in info, f'Info must have a address key, got {info.keys()}'
        return {'success': True, 'w': w}

    def eval_module(self, module:str, network=None):
        return c.gather([self.async_eval_module(module=module, network=network)])
    
    async def async_eval_module(self, module:str, network = None):
        """
        The following evaluates a module sver
        """
        # load the module stats (if it exists)
        
        if network != None:
            self.sync_network(network=network)

        module_info = self.load_module_info( module, {})
        namespace = self.namespace
        # CONFIGURE THE ADDRESS AND NAME (INFER THE NAME IF THE ADDDRESS IS PASED)
        if module in namespace:
            module_name = module
            module_address = namespace[module]
        else:
            module_address = module
            module_name = self.address2name.get(module_address, module_address)
        # emoji = c.emoji('hi')
        # CHECK IF THE MODULE IS EVALUATING ITSELF, DISABLE FOR NOW
        if not hasattr(self, 'my_info'):
            self.my_info = self.info()
        my_info = self.my_info
        if module_address == my_info['address']:
            return {'error': f'Cannot evaluate self {module_address}'}

        start_timestamp = c.time()


        self.requests += 1
        # BEGIN EVALUATION
        try:
            module = c.connect(module_address, key=self.key)
        except Exception as e:
            self.errors += 1
            e = c.detailed_error(e)
            return {'error': f'Error connecting to {module_address} {e}'}

        seconds_since_called = c.time() - module_info.get('timestamp', 0)

        # TEST IF THE MODULE IS WAS TESTED TOO RECENTLY
        if seconds_since_called > self.config.max_staleness :
            # c.print(f'{prefix} [bold yellow] {module["name"]} is too new as we pinged it {staleness}(s) ago[/bold yellow]', color='yellow')
            info = module.info(timeout=self.config.info_timeout)
            module_info.update(info)

        try:

            # check the info of the module
            # this is where we connect to the client
            response = self.score_module(module)
            response['msg'] = f'{c.emoji("checkmark")}{module_name} --> w:{response["w"]} {c.emoji("checkmark")} '
            self.successes += 1
        except Exception as e:
            e = c.detailed_error(e)
            self.errors += 1
            response = {
                        'w': 0, 
                        'msg': f'{c.emoji("cross")} {module_name} --> {e} {c.emoji("cross")}'  
                        }
            c.print(response['msg'], color='red', verbose=self.config.verbose)
            

        end_timestamp = c.time()        
        w = response['w']
        # we only want to save the module stats if the module was successful
        module_info['count'] = module_info.get('count', 0) + 1 # update the count of times this module was hit
        module_info['w'] = module_info.get('w', w)*(1-self.config.alpha) + w * self.config.alpha
        module_info['timestamp'] = end_timestamp
        module_info['start_timestamp'] = start_timestamp
        module_info['end_timestamp'] = end_timestamp
        module_info['latency'] = end_timestamp - start_timestamp

        history_record = {k:module_info[k] for k in self.config.history_features}
        module_info['history'] = (module_info.get('history', []) + [history_record])
        module_info['history'] = module_info['history'][:self.config.max_history]

        self.save_module_info(module_name, module_info)

        # VERBOSITY
        emoji = c.emoji('checkmark') if response['w'] > 0 else c.emoji('cross')
        
        
        c.print(f'{emoji} {module_name}:{module_address} --> {w} {emoji}', color='cyan', verbose=self.config.verbose)
        self.count += 1
        self.last_evaluation_time = c.time()
        return module_info

    @classmethod
    def resolve_storage_path(cls, network:str = 'subspace', tag:str=None):
        if tag == None:
            tag = 'base'
        return f'{tag}.{network}'
        
    def refresh_stats(self, network='subspace', tag=None):
        path = self.resolve_storage_path(network=network, tag=tag)
        return self.rm(path)
    
    def resolve_tag(self, tag:str=None):
        return self.tag if tag == None else tag
    
    def calculate_votes(self, tag=None, network = None):
        network = network or self.config.network
        tag = tag or self.tag

        # get the list of modules that was validated
        module_infos = self.module_infos(network=network, keys=['name','uid', 'w', 'ss58_address'], tag=tag)
        votes = {
            'keys' : [],            # get all names where w > 0
            'weights' : [],  # get all weights where w > 0
            'uids': [],
            'timestamp' : c.time()
        }

        key2uid = self.subspace.key2uid()
        for info in module_infos:
            if 'ss58_address' in info and info['w'] > 0:
                if info['ss58_address'] in key2uid:
                    votes['keys'] += [info['ss58_address']]
                    votes['weights'] += [info['w']]
                    votes['uids'] += [key2uid[info['ss58_address']]]

        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'

        return votes

    @property
    def last_vote_time(self):
        votes = self.load_votes()
        return votes.get('timestamp', 0)

    def load_votes(self) -> dict:
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


    def vote(self, tag=None, votes=None):

        votes = votes or self.calculate_votes(tag=tag) 
        if tag != None:
            key = self.resolve_server_name(tag=tag)
            key = c.get_key(key)
        else:
            key = self.key

        if len(votes['uids']) < self.config.min_num_weights:
            return {'success': False, 'msg': 'The votes are too low'}

        r = c.vote(uids=votes['uids'], # passing names as uids, to avoid slot conflicts
                        weights=votes['weights'], 
                        key=self.key, 
                        network=self.config.network, 
                        netuid=self.config.netuid)

        self.save_votes(votes)

        return {'success': True, 
                'message': 'Voted', 
                'votes': votes , 
                'r': r}

    @classmethod
    def module_names(cls, network:str=network, tag:str=None):
        paths = cls.module_paths(network=network, tag=tag)
        modules = [p.split('/')[-1].replace('.json', '') for p in paths]
        return modules

    @classmethod
    def num_module_infos(cls, tag=None, network=network):
        return len(cls.module_names(network=network,tag=tag))
        
    @classmethod
    def module_infos(cls,
                     tag=None,
                      network:str='subspace', 
                    batch_size:int=20 , # batch size for 
                    max_staleness:int= 1000,
                    keys:str=None):

        paths = cls.module_paths(network=network, tag=tag)   
        c.print(f'Loading {len(paths)} module infos', color='cyan')
        jobs = [c.async_get_json(p) for p in paths]
        module_infos = []


        # chunk the jobs into batches
        for jobs_batch in c.chunk(jobs, batch_size):
            results = c.gather(jobs_batch)
            # last_interaction = [r['history'][-1][] for r in results if r != None and len(r['history']) > 0]
            for s in results:
                if s == None:
                    continue
                if 'timestamp' in s:
                    s['staleness'] = c.timestamp() - s['timestamp']
                else:
                    s['staleness'] = 0
                if s['staleness'] > max_staleness:
                    continue
                if keys  != None:
                    s = {k: s.get(k,None) for k in keys}
                module_infos += [s]
        
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
    
    def save_module_info(self,k:str, v):
        path = self.resolve_storage_path(network=self.config.network, tag=self.tag) + f'/{k}'
        self.put_json(path, v)


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
        self = cls(**kwargs )
        return self.rufn()

    @classmethod
    def dashboard(cls):
        import streamlit as st
        # disable the run_loop to avoid the background  thread from running
        self = cls(start=False)
        module_path = self.path()
        network = 'local'
        c.new_event_loop()
        
        st.title(module_path)


        servers = c.servers(search='vali')
        server = st.selectbox('Select Vali', servers)
        state_path = f'dashboard/{server}'
        state = self.get(state_path, {})
        server = c.connect(server)
        update = st.button('Update')
        if len(state) == 0 or update:
            state = {
                'run_info': server.run_info(),
                'module_infos': server.module_infos()
            }

            self.put(state_path, state)


        run_info = state['run_info']
        module_infos = state['module_infos']
        df = []
        default_columns = ['name', 'staleness', 'w', 'count', 'timestamp']
        columns = list(module_infos[0].keys())
        selected_columns = st.multiselect('Select columns', columns, default_columns)
        search = st.text_input('Search')
        
        for row in module_infos:
            if search != '' and search not in row['name']:
                continue
            
            row = {k: row.get(k, None) for k in selected_columns}
            df += [row]
        
        df = c.df(df)

        st.write(df)
        st.write(run_info)


        
        vali_modules = c.my_modules(fmt='j', search='vali')
        vali_names =  [v['name'] for v in vali_modules]
        vali_name = st.selectbox('select vali', vali_names)
        
        module = vali_name.split("::")[0] if '::' in vali_name else vali_name
        tag = vali_name.split("::")[-1] if '::' in vali_name else None
        df = c.df(vali_modules)
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
