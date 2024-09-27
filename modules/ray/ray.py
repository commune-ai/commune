
import commune as c
from typing import *
import ray 
import json
import torch


class Ray(c.Module):
    description = 'ray stuff'
    def ray_context(self):
        return ray.runtime_context.get_runtime_context()
    
    def stop(self):
        return self.run_command('ray stop')
    
    def ray_put(self, *items):
        ray = self.env()
        return [ray.put(i) for i in items]
    

    
    def env(self):
        if not self.is_initialized():
            self.init()
        return ray
    

    
    def start(self):
        return c.cmd('ray start --head', verbose=True)
    

    @staticmethod
    def get_runtime_context():
        return ray.runtime_context.get_runtime_context()
    

    
    def ensure_ray_context(self, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if self.is_initialized():
            ray_context = self.get_runtime_context()
        else:
            ray_context =  self.init(init_kwargs=ray_config)
        
        return ray_context
    

     
    def ray_launch(self, 
                   module= None, 
                   name:Optional[str]=None, 
                   tag:str=None, 
                   args:List = None, 
                   refresh:bool = False,
                   kwargs:Dict = None,
                   serve: bool = False, 
                   **actor_kwargs):
        
        launch_kwargs = dict(locals())
        launch_kwargs.update(launch_kwargs.pop('actor_kwargs'))
        ray = self.env()
        """
        deploys process as an actor or as a class given the config (config)
        """
        args = args if args != None else []
        kwargs = kwargs if kwargs != None else {}
        module_class = None
        if isinstance(module, str):
            module_class = self.get_module(module)
        elif module == None :
            module_class = self
    
        else:
            module_class = c.module(module)
            
        name = self.get_server_name(name=name, tag=tag) 
        assert isinstance(name, str)
        
        actor_kwargs['name'] = name
        actor_kwargs['refresh'] = refresh
    
        actor = self.create_actor(module=module_class,  args=args, kwargs=kwargs, **actor_kwargs) 
        if serve:
            actor = actor.serve(ray_get=False)
        
        return actor
    

    
    def ray_runtime_context(self):
        return ray.get_runtime_context()
    

    
    def restart(self, stop:dict={}, start:dict={}):
        '''
        
        Restart  ray cluster
        
        '''
        command_out_dict = {}
        command_out_dict['stop'] = self.stop(**stop)
        command_out_dict['start'] = self.start(**start)
        return command_out_dict
    

    
    def kill_actor(self, actor, verbose=True):
        
    
        if self.actor_exists(actor):
            actor = ray.get_actor(actor)
        else:
            if verbose:
                print(f'{actor} does not exist for it to be removed')
            return None
        ray.kill(actor)
    
        return True
    

    @staticmethod
    def ray_nodes( *args, **kwargs):
        from ray.experimental.state.api import list_nodes
        return list_nodes(*args, **kwargs)
    

    
    def ray_import(self):
        
        return ray
    

    
    def ray_namespace(self):
        
        return ray.get_runtime_context().namespace
    

    
    def ray_wait(self, *jobs):
        self.env()
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs
    

    
    def actors(self, *args, **kwargs):
        ray = self.env()
        actors =  self.list_actors(*args, **kwargs)
        actors = [a['name'] for a in actors] 
        return actors
    
    
    def list_actors(self, *args, **kwargs):
        from ray.util.state import list_actors
        return list_actors(*args, **kwargs)
    
    def actor2id(self, actor_name:str = None):
        actors = self.list_actors()
        actor2id =  {a['name']:a['actor_id'] for a in actors}
        if actor_name:
            return actor2id[actor_name]
        return actor2id
    
    def id2actor(self, actor_id:str = None):
        actors = self.list_actors()
        id2actor =  {a['actor_id']:a['name'] for a in actors}
        if actor_id:
            return id2actor[actor_id]
        return id2actor
    
    


    


    
    def is_initialized(self):
        
        return ray.is_initialized()
    

    
    def status(self, *args, **kwargs):
        return self.run_command('ray status',  *args, **kwargs)
    

    
    def ray_get(self,*jobs):
        self.env()
        return ray.get(jobs)
    

    
    def ray_tasks(self, running=False, name=None, *args, **kwargs):
        ray = self.env()
        filters = []
        if running == True:
            filters.append([("scheduling_state", "=", "RUNNING")])
        if isinstance(name, str):
            filters.append([("name", "=", name)])
        
        if len(filters)>0:
            kwargs['filters'] = filters
    
        ray_tasks = ray.experimental.state.api.list_tasks(*args, **kwargs)
        return ray_tasks
    


    @staticmethod
    def list_objects( *args, **kwargs):
        return ray.experimental.state.api.list_objects(*args, **kwargs)
    

    
    def ray_actor_map(self, ):
        ray = self.env()
        actor_list = self.actors(names_only=False, detail=True)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map
    
    
    def resolve_actor_id(self, actor_id:str, **kwargs):
        actor2id = self.actor2id(**kwargs)
        if actor_id in actor2id:
            actor_id = actor2id[actor_id]
        return actor_id
    
    
    def get_logs(self, actor_id:str, **kwargs):
        from ray.util.state import get_log
        actor_id = self.resolve_actor_id(actor_id)
        return get_log(actor_id=actor_id, **kwargs)

    
    def get_actor(self ,actor_name:str, virtual:bool=False):
        '''
        Gets the ray actor
        '''
        ray  = self.env()
        actor =  ray.get_actor(actor_name)
        # actor = Module.add_actor_metadata(actor)
        # if virtual:
        #     actor = self.virtual_actor(actor=actor)
        return actor
    

    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'}
    
    def init(self,init_kwargs={}):
        
        init_kwargs =  {**self.default_ray_env, **init_kwargs}
        ray_context = {}
        is_initialized = self.is_initialized()
        c.print(f'RAY is_initialized: {is_initialized}')


        if  is_initialized:
             ray_context =  self.ray_runtime_context()
        else: 
            ray_context = ray.init(**init_kwargs)
        c.print(f'CONNECTED TO RAY: {ray_context}')
        return ray_context
    
    
    def serve(self,
                 module : str = None,
                 name:str = None,
                 kwargs: dict = None,
                 args:list =None,
                 cpus:int = 1.0,
                 gpus:int = 0,
                 detached:bool=True, 
                 max_concurrency:int=50,
                 namespace = 'default',
                 refresh:bool=True,
                 virtual:bool = True):
        
        self.init()

        if isinstance(module, str):
            name = module or 'module'
            module = c.module(module)

        name = name if name != None else module.__name__

        self_kwargs = kwargs if kwargs else {}
        self_args = args if args else []
        resources = {}
        resources['num_cpus'] = cpus
        resources['num_gpus'] = gpus

        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs
        options_kwargs = {'name': name,
                          'max_concurrency': max_concurrency,
                           **resources}
        
        # detatch the actor from the process when it finishes
        if detached:
            options_kwargs['lifetime'] = 'detached'
            
        # setup class init config
        # refresh the actor by killing it and starting it (assuming they have the same name)
        if refresh:
            if self.actor_exists(name):
                self.kill_actor(actor=name,verbose=True)
                # assert not Module.actor_exists(name)


        options_kwargs['namespace'] = namespace
        actor = ray.remote(**options_kwargs)(module).remote(*self_args, **self_kwargs)

        # create the actor if it doesnt exisst
        # if the actor is refreshed, it should not exist lol (TODO: add a check)
        
        actor = self.get_actor(name, virtual=virtual)

        return actor

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()


    
    def virtual_actor(self, actor):
        return c.module('ray.client')(actor=actor)

    
    def kill_actor(self, actor, verbose=True):
        

        if self.actor_exists(actor):
            actor = ray.get_actor(actor)
        else:
            if verbose:
                print(f'{actor} does not exist for it to be removed')
            return None
        ray.kill(actor)
    
        return True
    kill = kill_actor
        
       
    
    def actor_exists(self, actor):
        ray = self.env()
        if isinstance(actor, str):
            try:
                ray.get_actor(actor)
                actor_exists = True
            except ValueError as e:
                actor_exists = False
            
            return actor_exists
        else:
            raise NotImplementedError

    
    def actor(self ,actor_name:str, virtual:bool=True, **kwargs):
        '''
        Gets the ray actor
        '''
        ray  = self.env()
        actor =  ray.get_actor(actor_name, **kwargs)
        # actor = Module.add_actor_metadata(actor)
        # if virtual:
        #     actor = self.virtual_actor(actor=actor)
        return actor

        
    def call(self, module, fn = None, *args, **kwargs):
        if '/' in module:
            module, fn = module.split('/')
            args = [fn] + list(args)
        
        return self.get_actor(module).remote(fn)(*args, **kwargs)
    get_actor = get_actor

    
    def ray_runtime_context(self):
        
        return ray.get_runtime_context()

    
    def ray_namespace(self):
        
        return ray.get_runtime_context().namespace

    
    def ray_context(self):
        
        
        return ray.runtime_context.get_runtime_context()

    @staticmethod
    def ray_objects( *args, **kwargs):
        
        return ray.experimental.state.api.list_objects(*args, **kwargs)
    

    
    def actor_resources(self, actor:str):
        resource_map = self.ray_actor_map()[actor]['required_resources']
        k_map = {
            'GPU': 'gpus',
            'CPU': 'cpus'
        }
        return {k_map[k]:float(v) for k,v in resource_map.items() }
    
    def ray_actor_map(self, ):
        ray = self.env()
        actor_list = self.actors(names_only=False, detail=True)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map
    actor_map = ray_actor_map
    
    
    def ray_tasks(self, running=False, name=None, *args, **kwargs):
        ray = self.env()
        filters = []
        if running == True:
            filters.append([("scheduling_state", "=", "RUNNING")])
        if isinstance(name, str):
            filters.append([("name", "=", name)])
        
        if len(filters)>0:
            kwargs['filters'] = filters

        ray_tasks = ray.experimental.state.api.list_tasks(*args, **kwargs)
        return ray_tasks
   
    @staticmethod
    def ray_nodes( *args, **kwargs):
        from ray.experimental.state.api import list_nodes
        return list_nodes(*args, **kwargs)
    
    def ray_get(self,*jobs):
        self.env()
        return ray.get(jobs)
    
    def ray_wait(self, *jobs):
        self.env()
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs
    
    
    
    def ensure_ray_context(self, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if self.is_initialized():
            ray_context = self.get_runtime_context()
        else:
            ray_context =  self.init(init_kwargs=ray_config)
        
        return ray_context


    
    def ray_put(self, *items):
        ray = self.env()
        
        return [ray.put(i) for i in items]

    @staticmethod
    def get_runtime_context():
        
        return ray.runtime_context.get_runtime_context()
    

    ## RAY LAND
    
    def stop(self):
        """
        stop cluster
        """
        c.cmd('ray stop')
    
    def start(self):
        '''
        Start the ray cluster 
        '''
        return c.cmd('ray start --head')

    
    def restart(self, stop:dict={}, start:dict={}):
        '''
        
        Restart  ray cluster
        
        '''
        command_out_dict = {}
        command_out_dict['stop'] = self.stop(**stop)
        command_out_dict['start'] = self.start(**start)
        return command_out_dict


    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'
                      
                      }

    # def resource_usage(self):
    #     resource_dict =  self.config.get('actor', {}).get('resources', None)
    #     resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
    #     resource_dict['memory'] = self.memory_usage(mode='ratio')
    #     return  resource_dict
    
