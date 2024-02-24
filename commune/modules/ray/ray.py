
import commune as c
from typing import *
import ray 
import json
import torch


class Ray(c.Module):
    description = 'ray stuff'
    @classmethod
    def ray_context(cls):
        
        
        return ray.runtime_context.get_runtime_context()
    
    @classmethod
    def stop(cls):
        return cls.run_command('ray stop')
    

    @classmethod
    def ray_put(cls, *items):
        ray = cls.env()
        return [ray.put(i) for i in items]
    

    @classmethod
    def env(cls):
        if not cls.is_initialized():
            cls.init()
        return ray
    

    @classmethod
    def start(cls):
        return c.cmd('ray start --head', verbose=True)
    

    @staticmethod
    def get_runtime_context():
        return ray.runtime_context.get_runtime_context()
    

    @classmethod
    def ensure_ray_context(cls, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if cls.is_initialized():
            ray_context = cls.get_runtime_context()
        else:
            ray_context =  cls.init(init_kwargs=ray_config)
        
        return ray_context
    

    @classmethod 
    def ray_launch(cls, 
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
        ray = cls.env()
        """
        deploys process as an actor or as a class given the config (config)
        """
        args = args if args != None else []
        kwargs = kwargs if kwargs != None else {}
        module_class = None
        if isinstance(module, str):
            module_class = cls.get_module(module)
        elif module == None :
            module_class = cls
    
        else:
            module_class = c.module(module)
            
        name = self.get_server_name(name=name, tag=tag) 
        assert isinstance(name, str)
        
        actor_kwargs['name'] = name
        actor_kwargs['refresh'] = refresh
    
        actor = cls.create_actor(module=module_class,  args=args, kwargs=kwargs, **actor_kwargs) 
        if serve:
            actor = actor.serve(ray_get=False)
        
        return actor
    

    @classmethod
    def ray_runtime_context(cls):
        return ray.get_runtime_context()
    

    @classmethod
    def restart(cls, stop:dict={}, start:dict={}):
        '''
        
        Restart  ray cluster
        
        '''
        command_out_dict = {}
        command_out_dict['stop'] = cls.stop(**stop)
        command_out_dict['start'] = cls.start(**start)
        return command_out_dict
    

    @classmethod
    def kill_actor(cls, actor, verbose=True):
        
    
        if cls.actor_exists(actor):
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
    

    @classmethod
    def ray_import(cls):
        
        return ray
    

    @classmethod
    def ray_namespace(cls):
        
        return ray.get_runtime_context().namespace
    

    @classmethod
    def ray_wait(cls, *jobs):
        cls.env()
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs
    

    @classmethod
    def actors(cls, *args, **kwargs):
        ray = cls.env()
        actors =  cls.list_actors(*args, **kwargs)
        actors = [a['name'] for a in actors] 
        return actors
    
    @classmethod
    def list_actors(cls, *args, **kwargs):
        from ray.util.state import list_actors
        return list_actors(*args, **kwargs)
    @classmethod
    def actor2id(cls, actor_name:str = None):
        actors = cls.list_actors()
        actor2id =  {a['name']:a['actor_id'] for a in actors}
        if actor_name:
            return actor2id[actor_name]
        return actor2id
    
    def id2actor(cls, actor_id:str = None):
        actors = cls.list_actors()
        id2actor =  {a['actor_id']:a['name'] for a in actors}
        if actor_id:
            return id2actor[actor_id]
        return id2actor
    
    


    


    @classmethod
    def is_initialized(cls):
        
        return ray.is_initialized()
    

    @classmethod
    def status(cls, *args, **kwargs):
        return cls.run_command('ray status',  *args, **kwargs)
    

    @classmethod
    def ray_get(cls,*jobs):
        cls.env()
        return ray.get(jobs)
    

    @classmethod
    def ray_tasks(cls, running=False, name=None, *args, **kwargs):
        ray = cls.env()
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
    

    @classmethod
    def ray_actor_map(cls, ):
        ray = cls.env()
        actor_list = cls.actors(names_only=False, detail=True)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map
    
    @classmethod
    def resolve_actor_id(cls, actor_id:str, **kwargs):
        actor2id = cls.actor2id(**kwargs)
        if actor_id in actor2id:
            actor_id = actor2id[actor_id]
        return actor_id
    
    @classmethod
    def get_logs(cls, actor_id:str, **kwargs):
        from ray.util.state import get_log
        actor_id = cls.resolve_actor_id(actor_id)
        return get_log(actor_id=actor_id, **kwargs)

    @classmethod
    def get_actor(cls ,actor_name:str, virtual:bool=False):
        '''
        Gets the ray actor
        '''
        ray  = cls.env()
        actor =  ray.get_actor(actor_name)
        # actor = Module.add_actor_metadata(actor)
        # if virtual:
        #     actor = cls.virtual_actor(actor=actor)
        return actor
    

    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'}
    @classmethod
    def init(cls,init_kwargs={}):
        
        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        ray_context = {}
        is_initialized = cls.is_initialized()
        c.print(f'RAY is_initialized: {is_initialized}')


        if  is_initialized:
             ray_context =  cls.ray_runtime_context()
        else: 
            ray_context = ray.init(**init_kwargs)
        c.print(f'CONNECTED TO RAY: {ray_context}')
        return ray_context
    
    @classmethod
    def serve(cls,
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
        
        cls.init()

        if isinstance(module, str):
            name = module or 'module'
            module = c.module(module)

        name = name if name != None else module.__name__

        cls_kwargs = kwargs if kwargs else {}
        cls_args = args if args else []
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
            if cls.actor_exists(name):
                cls.kill_actor(actor=name,verbose=True)
                # assert not Module.actor_exists(name)


        options_kwargs['namespace'] = namespace
        actor = ray.remote(**options_kwargs)(module).remote(*cls_args, **cls_kwargs)

        # create the actor if it doesnt exisst
        # if the actor is refreshed, it should not exist lol (TODO: add a check)
        
        actor = cls.get_actor(name, virtual=virtual)

        return actor

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()


    @classmethod
    def virtual_actor(cls, actor):
        return c.module('ray.client')(actor=actor)

    @classmethod
    def kill_actor(cls, actor, verbose=True):
        

        if cls.actor_exists(actor):
            actor = ray.get_actor(actor)
        else:
            if verbose:
                print(f'{actor} does not exist for it to be removed')
            return None
        ray.kill(actor)
    
        return True
    kill = kill_actor
        
       
    @classmethod
    def actor_exists(cls, actor):
        ray = cls.env()
        if isinstance(actor, str):
            try:
                ray.get_actor(actor)
                actor_exists = True
            except ValueError as e:
                actor_exists = False
            
            return actor_exists
        else:
            raise NotImplementedError

    @classmethod
    def actor(cls ,actor_name:str, virtual:bool=True, **kwargs):
        '''
        Gets the ray actor
        '''
        ray  = cls.env()
        actor =  ray.get_actor(actor_name, **kwargs)
        # actor = Module.add_actor_metadata(actor)
        # if virtual:
        #     actor = cls.virtual_actor(actor=actor)
        return actor

    @classmethod    
    def call(cls, module, fn = None, *args, **kwargs):
        if '/' in module:
            module, fn = module.split('/')
            args = [fn] + list(args)
        
        return cls.get_actor(module).remote(fn)(*args, **kwargs)
    get_actor = get_actor

    @classmethod
    def ray_runtime_context(cls):
        
        return ray.get_runtime_context()

    @classmethod
    def ray_namespace(cls):
        
        return ray.get_runtime_context().namespace

    @classmethod
    def ray_context(cls):
        
        
        return ray.runtime_context.get_runtime_context()

    @staticmethod
    def ray_objects( *args, **kwargs):
        
        return ray.experimental.state.api.list_objects(*args, **kwargs)
    

    @classmethod
    def actor_resources(cls, actor:str):
        resource_map = cls.ray_actor_map()[actor]['required_resources']
        k_map = {
            'GPU': 'gpus',
            'CPU': 'cpus'
        }
        return {k_map[k]:float(v) for k,v in resource_map.items() }
    @classmethod
    def ray_actor_map(cls, ):
        ray = cls.env()
        actor_list = cls.actors(names_only=False, detail=True)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map
    actor_map = ray_actor_map
    
    @classmethod
    def ray_tasks(cls, running=False, name=None, *args, **kwargs):
        ray = cls.env()
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
    @classmethod
    def ray_get(cls,*jobs):
        cls.env()
        return ray.get(jobs)
    @classmethod
    def ray_wait(cls, *jobs):
        cls.env()
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs
    
    
    @classmethod
    def ensure_ray_context(cls, ray_config:dict = None):
        ray_config = ray_config if ray_config != None else {}
        
        if cls.is_initialized():
            ray_context = cls.get_runtime_context()
        else:
            ray_context =  cls.init(init_kwargs=ray_config)
        
        return ray_context


    @classmethod
    def ray_put(cls, *items):
        ray = cls.env()
        
        return [ray.put(i) for i in items]

    @staticmethod
    def get_runtime_context():
        
        return ray.runtime_context.get_runtime_context()
    

    ## RAY LAND
    @classmethod
    def stop(cls):
        cls.run_command('ray stop')


    @classmethod
    def start(cls):
        '''
        Start the ray cluster 
        (TODO: currently supports head)
        '''
        return c.cmd('ray start --head')

    @classmethod
    def restart(cls, stop:dict={}, start:dict={}):
        '''
        
        Restart  ray cluster
        
        '''
        command_out_dict = {}
        command_out_dict['stop'] = cls.stop(**stop)
        command_out_dict['start'] = cls.start(**start)
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
    
