import torch
import traceback
import commune as c
import concurrent


class Vali(c.Module):
    
    last_sync_time = 0

    def __init__(self, config=None, **kwargs):
        self.init_vali(config=config, **kwargs)

    def init_vali(self, config=None, **kwargs):
        config = self.set_config(config=config, kwargs=kwargs)
        # merge the config with the default config
        self.count = 0

        # we want to make sure that the config is a munch
        self.config = c.munch({**Vali.config(), **config})
        self.start_time = c.time()
        self.errors = 0
        c.print(f'Vali config: {self.config}', color='cyan')
        if self.config.start:
            self.sync()
            self.executor = c.module('thread.pool')(num_workers=self.config.num_workers, save_outputs=False)
            c.thread(self.run)
            c.thread(self.vote_loop)
    
    @property
    def sync_staleness(self):
        return int(c.time() - self.last_sync_time) 

    def vote_loop(self):
        while True:
            c.sleep(1)
            if self.vote_staleness > self.config.vote_interval:
                try:
                    c.print('Voting...', color='cyan')
                    self.vote()
                except Exception as e:
                    c.print(f'Error voting {e}', color='red')
                    c.print(traceback.format_exc(), color='red')
                    c.sleep(1)
                    continue


    @property               
    def search(self):
        if self.config.search == None:
            self.config.search = self.tag
        assert isinstance(self.config.search, str), f'Module search must be a string, got {type(self.config.search)}'
        return self.config.search


    def sync(self, network:str=None, netuid:int=None, update: bool = False):
        
        try:
            if network == None:
                network = self.config.network
            if netuid == None:
                netuid = self.config.netuid
            self.subspace = c.module('subspace')(network=network, netuid=netuid)
            
            self.modules = self.subspace.modules(search=self.config.search, update=update, netuid=netuid)
            self.n  = len(self.modules)                
            self.subnet = self.subspace.subnet(netuid=netuid)

            if self.config.vote_interval == None: 
                self.config['vote_interval'] = self.subspace.seconds_per_epoch()

            self.last_sync_time = c.time()
            self.block = self.subspace.block

            c.print('Syncing...', color='cyan')
        except Exception as e:
            c.print(f'Error syncing {e}', color='red')
            c.print(traceback.format_exc(), color='red')
            return {'success': False, 'message': f'Error syncing {e}'}

        return {'modules': self.modules, 'subnet': self.subnet}

    def score_module(self, module):

        '''
        params:
            module: module client
            kwargs : the key word arguments
        
        '''

        info = module.info()
        assert isinstance(info, dict), f'Response must be a dict, got {type(info)}'
        assert 'address' in info, f'Response must have an error key, got {info.keys()}'
        return {'success': True, 'w': 1}


    def eval_module(self, module:dict):
        """
        The following evaluates a module server, from the dictionary
        """
        
        # load the module stats (if it exists)
        module_stats = self.load_module_stats( module['name'], default=module)

        # update the module state with the module stats
        module_stats.update(module)
        
        staleness = c.time() - module_stats.get('timestamp', 0)
        if staleness < self.config.max_staleness:
            # c.print(f'{prefix} [bold yellow] {module["name"]} is too new as we pinged it {staleness}(s) ago[/bold yellow]', color='yellow')
            return {'error': f'{module["name"]} is too new as we pinged it {staleness}(s) ago'}

        try:
            # this is where we connect to the client
            module_client = c.connect(module['address'], key=self.key, virtual=True)
            response = self.score_module(module_client)
            msg = f'{c.emoji("check")}{module["name"]} --> w:{response["w"]} {c.emoji("check")} '
            color = 'green'

        except Exception as e:
            msg = f'{c.emoji("cross")} {module["name"]} {e} {c.emoji("cross")}'  
            response = {'error': c.detailed_error(e), 'w': 0}
            color = 'red'

        c.print(msg, color=color)
        c.print(response)
        
        self.count += 1

        w = response['w']
        response['timestamp'] = c.time()
        # we only want to save the module stats if the module was successful
        module_stats['count'] = module_stats.get('count', 0) + 1 # update the count of times this module was hit
        module_stats['w'] = module_stats.get('w', w)*(1-self.config.alpha) + w * self.config.alpha
        module_stats['timestamp'] = response['timestamp']

        # add the history of this module
        module_stats['history'] = module_stats.get('history', []) + [response]
        module_stats['history'] = module_stats['history'][-self.config.max_history:]
        self.save_module_stats(module['name'], module_stats)

        return module_stats

    @classmethod
    def networks(cls):
        return [f.split('/')[-1] for f in cls.ls('stats')]

    @classmethod
    def resolve_stats_path(cls, network:str, tag:str=None):
        if tag == None:
            tag = 'base'
        return f'stats/{network}/{tag}'
        
    def refresh_stats(self, network='main', tag=None):
        tag = self.tag if tag == None else tag
        path = self.resolve_stats_path(network=network, tag=tag)
        return self.rm(path)
    
    def resolve_tag(self, tag:str=None):
        return self.tag if tag == None else tag



    def votes(self, network='main', tag=None):
        tag = self.resolve_tag(tag)
        stats = self.module_stats(network=network, keys=['name','uid', 'w'], tag=tag)

        votes = {
            'names'     : [v['name'] for v in stats],            # get all names where w > 0
            'uids'      : [v['uid'] for v in stats],             # get all uids where w > 0
            'weights'   : [v['w'] for v in stats],  # get all weights where w > 0
            'timestamp' : c.time()
        }
        assert len(votes['uids']) == len(votes['weights']), f'Length of uids and weights must be the same, got {len(votes["uids"])} uids and {len(votes["weights"])} weights'
        
        
        c.copy(votes['uids']) # is line needed ?
        new_votes = {'names': [], 'uids': [], 'weights': [], 'timestamp': c.time(), 'block': self.block}
        for i in range(len(votes['names'])):
            if votes['uids'][i] < self.n :
                new_votes['names'] += [votes['names'][i]]
                new_votes['uids'] += [votes['uids'][i]]
                new_votes['weights'] += [votes['weights'][i]]
        
        votes = new_votes
        topk = self.subnet['max_allowed_weights']
        topk_indices = torch.argsort( torch.tensor(votes['weights']), descending=True)[:topk].tolist()
        votes['weights'] = [votes['weights'][i] for i in topk_indices]
        votes['names'] = [votes['names'][i] for i in topk_indices]
        # normalize vote
        votes['weights'] = torch.tensor(votes['weights'])
        votes['weights'] = (votes['weights'] / votes['weights'].sum())
        votes['weights'] = votes['weights'].tolist()
        return votes

    def vote(self):
        c.print(f'Voting on {self.config.network} {self.config.netuid}', color='cyan')
        stake = self.subspace.get_stake(self.key.ss58_address, netuid=self.config.netuid)

        if stake < self.config.min_stake:
            result = {'success': False, 'message': f'Not enough  {self.key.ss58_address} ({self.key.path}) stake to vote, need at least {self.config.min_stake} stake'}
            c.print(result, color='red')
            return result

        # calculate votes
        votes = self.votes()

        c.print(f'Voting on {len(votes["names"])} modules', color='cyan')
        self.subspace.vote(uids=votes['names'], # passing names as uids, to avoid slot conflicts
                        weights=votes['weights'], 
                        key=self.key, 
                        network=self.config.network, 
                        netuid=self.config.netuid)

        self.save_votes(votes)

        return {'success': True, 'message': 'Voted', 'votes': votes }

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
        self.put(f'votes/{self.config.network}/{self.tag}', votes)

    @classmethod
    def tags(cls, network='main', mode='stats'):
        return list(cls.tag2path(network=network, mode=mode).keys())

    @classmethod
    def paths(cls, network='main', mode='stats'):
        return list(cls.tag2path(network=network, mode=mode).values())

    @classmethod
    def tag2path(cls, network:str='main', mode='stats'):
        return {f.split('/')[-1].split('.')[0]: f for f in cls.ls(f'{mode}/{network}')}

    @classmethod
    def sand(cls):
        for path in cls.ls('votes'):
            if '/main.' in path:
                new_path = c.copy(path)
                new_path = new_path.replace('/main.', '/main/')
                c.mv(path, new_path)

    @classmethod
    def saved_module_paths(cls, network:str='main', tag:str=None):
        tag = 'base' if tag == None else tag
        paths = cls.ls(f'stats/{network}/{tag}')
        return paths

    @classmethod
    def saved_module_names(cls, network:str='main', tag:str=None):
        paths = cls.saved_module_paths(network=network, tag=tag)
        modules = [p.split('/')[-1].replace('.json', '') for p in paths]
        return modules
        
    @classmethod
    def module_stats(cls,
                     tag=None,
                      network:str='main', 
                    batch_size:int=20 , 
                    max_staleness:int= 1000,
                    keys:str=None):

        paths = cls.saved_module_paths(network=network, tag=tag)   
        jobs = [c.async_get_json(p) for p in paths]
        module_stats = []

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
                module_stats += [s]
        
        return module_stats
    

    def ls_stats(self):
        paths = self.ls(f'stats/{self.config.network}')
        return paths

    def load_module_stats(self, k:str,default=None):
        default = default if default != None else {}
        path = self.resolve_stats_path(network=self.config.network, tag=self.tag) + f'/{k}'
        return self.get_json(path, default=default)


    def get_history(self, k:str, default=None):
        module_stats = self.load_module_stats(k, default=default)
        return module_stats.get('history', [])
    
    def save_module_stats(self,k:str, v):
        path = self.resolve_stats_path(network=self.config.network, tag=self.tag) + f'/{k}'
        self.put_json(path, v)


    @property
    def vote_staleness(self) -> int:
        return int(c.time() - self.last_vote_time)


    def run(self, vote=False):

        self.sync()
        if self.config.check_loop:
            self.ensure_check_loop()
        if self.config.refresh_stats:
            self.refresh_stats(network=self.config.network, tag=self.tag)
        c.print(f'Running -> network:{self.config.network} netuid: {self.config.netuid}', color='cyan')
        c.new_event_loop()
        self.running = True
        futures = []
        while self.running:

            modules = c.shuffle(c.copy(self.modules))
            time_between_interval = c.time()
            module = c.choice(modules)

            c.print(f'Sending -> {module["name"]} {c.emoji("rocket")} ({module["address"]}) {c.emoji("rocket")}', color='yellow')
            c.sleep(self.config.sleep_time)

            future = self.executor.submit(fn=self.eval_module, kwargs={'module':module}, return_future=True)
            futures.append(future)

            if len(futures) >= self.config.max_futures:
                for future in c.as_completed(futures, timeout=self.config.timeout):
                    result = future.result()
                    futures.remove(future)
                    break
            
            # complete the futures as they come in
            if self.sync_staleness > self.config.sync_interval:
                self.sync()

            if self.count % 10 == 0 and self.count > 0:
                stats =  {
                'total_modules': self.count,
                'lifetime': int(self.lifetime),
                'modules_per_second': int(self.modules_per_second()), 
                'vote_staleness': self.vote_staleness,
                'errors': self.errors,
                'vote_interval': self.config.vote_interval,
                'epochs': self.epochs,
                    }
                c.print(f'STATS  --> {stats}\n', color='white')


    @property
    def epochs(self):
        return self.count // self.n
    
           
    def check_score(self, module):
        module_name = module['name']

        return self.w.get(module_name, 0)
            
    def stop(self):
        self.running = False
        
    @classmethod
    def check_valis(cls, network='main', interval:int = 20, max_staleness:int=300, return_all=True, remote=False):
        # get the up to date vali stats
        vali_stats = cls.stats(network=network, df=False, return_all=return_all, update=True)
        for v in vali_stats:
            if 'serving' not in v:
                continue
            if v['staleness'] > max_staleness:
                c.print(f'{v["name"]} is stale {v["staleness"]}s, restrting', color='red')
                c.serve(v['name'])
            if v['serving'] == False:
                c.print(f'{v["name"]} is not serving, restrting', color='red')
                address = c.get_address(v['name'])
                port = None
                if address != None:
                    port = int(address.split(':')[-1])
                c.serve(v['name'], port=port)

            c.print(f'{interval} ', color='green')
            c.sleep(interval)

            

    check_loop_name = 'vali::check_loop'
    @classmethod
    def check_loop(cls, interval=2, remote=True, **kwargs):
        if remote:
            kwargs['remote'] = False
            cls.remote_fn('check_loop', name=cls.check_loop_name, kwargs=kwargs)
            return {'success': True, 'message': 'Started check_vali_loop'}
        while True:
            c.print('Checking valis', color='cyan')
            c.print(cls.all_stats())
            cls.check_valis(**kwargs)
            c.sleep(interval)

    @classmethod
    def check_loop_running(cls):
        return c.pm2_exists(cls.check_loop_name)

    @classmethod
    def ensure_check_loop(self):
        if self.check_loop_running() == False:
            self.check_loop(remote=True)

    # @classmethod
    # def stake_spread(cls, modulenetwork='main'):
    #     subspace = c.module('subspace')(network=network)
    #     total_stake = self.subspace.total_stake(netuid=self.config.netuid)
    #     return stake / total_stake

    @classmethod
    def stats(cls,     
                    network='main', 
                    df:bool = True,
                    sortby:str=['name'], 
                    update:bool=True, 
                    cache_path:str = 'vali_stats',
                    return_all:bool=False):
        if return_all:
            return cls.all_stats(network=network, df=df)
        vali_stats = []
        if update == False:
            vali_stats = cls.get(cache_path, default=[])


        if len(vali_stats) == 0:
            module_path = cls.module_path()
            stats = c.stats(module_path+'::', df=False, network=network)
            name2stats = {s['name']: s for s in stats}
            for tag, path in cls.tag2path(mode='votes', network=network).items():
                v = cls.get(path)
                name = module_path + "::" +tag
                vote_info = name2stats.get(name, {})

                if vote_info.get('registered') == None:
                    continue
                if 'timestamp' in v:
                    vote_info['name'] = name
                    vote_info['n'] = len(v['uids'])
                    vote_info['timestamp'] = v['timestamp']
                    vote_info['avg_w'] = sum(v['weights']) / (len(v['uids']) + 1e-8)
                    vali_stats += [vote_info]
            cls.put(cache_path, vali_stats)    

        
        for v in vali_stats:
            v['staleness'] = int(c.time() - v['timestamp'])

            del v['timestamp']


        if df:
            vali_stats = c.df(vali_stats)
            # filter out NaN values for registered modules
            # include nans  for registered modules
            if len(vali_stats) > 0:
                vali_stats.sort_values(sortby, ascending=False, inplace=True)
            


        
        
        return vali_stats


    @classmethod
    def all_stats(cls, network='main', df:bool = True, sortby:str=['name'] , update=True, cache_path:str = 'vali_stats'):
        modules = c.modules('vali')
        all_vote_stats = []
        for m in modules:
            if not m.startswith('vali'):
                continue 
            try:
                # WE ONLY WANT TO UPDATE THE STATS IF THE MODULE IS RUNNING
                m_vote_stats = c.module(m).stats(df=False, network=network, return_all=False, update=update)
                c.print(f'Got vote stats for {m} (n={len(m_vote_stats)})')
                if len(m_vote_stats) > 0:
                    all_vote_stats += m_vote_stats
            except Exception as e:
                e = c.detailed_error(e)
                c.print(c.dict2str(e), color='red')
                continue
                
        if df == True:
            df =  c.df(all_vote_stats)
            # filter out NaN values for registered modules
            df.sort_values(sortby, ascending=False, inplace=True)
            return df
        return all_vote_stats


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
        return self.run()


