import ray
import torch


# def limit_parallel_jobs()

def limit_parallel_wait(running_job_handles,
                        parallel_limit=2,
                        remote_fn=None,
                        fn_kwargs={}):
    finished_job_handles = []
    while True:
        if len(running_job_handles) > 0:
            finished_job_handles, running_job_handles = ray.wait(running_job_handles)
        else:
            finished_job_handles, running_job_handles = [], []

        if len(running_job_handles) <=parallel_limit:
            running_job_handles.append(remote_fn.remote(**fn_kwargs))
            break
    return finished_job_handles, running_job_handles


def kill_actor(actor, verbose=True):

    if isinstance(actor, str):
        if actor_exists(actor):
            actor = ray.get_actor(actor)
        else:
            if verbose:
                print(f'{actor} does not exist for it to be removed')
            return None

    return ray.kill(actor)



def actor_exists(actor):
    exists_bool = False
    
    try:
        ray.get_actor(actor)
        exists_bool = True
    except ValueError:
        exists_bool = False

    return exists_bool

def create_actor(cls,
                 name,
                 cls_kwargs,
                 detached=True,
                 resources={'num_cpus': 0.5, 'num_gpus': 0},
                 max_concurrency=5,
                 refresh=False,
                 return_actor_handle=False,
                 verbose = True,
                 redundant=False):
        '''
          params:
              config: configuration of the experiment
              run_dag: run the dag
              token_pairs: token pairs
              resources: resources per actor
              actor_prefix: prefix for the data actors
          '''
        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs

        options_kwargs = {'name': name,
                          'max_concurrency': max_concurrency,
                           **resources}
        if detached:
            options_kwargs['lifetime'] = 'detached'

        # setup class init config


        # refresh the actor by killing it and starting it (assuming they have the same name)
        if refresh:
            if actor_exists(name):
                kill_actor(actor=name,verbose=verbose)

        if redundant:
            # if the actor already exists and you want to create another copy but with an automatic tag
            actor_index = 0
            while not actor_exists(name):
                name =  f'{name}-{actor_index}' 
                actor_index += 1


        if not actor_exists(name):
            
            try:
                actor_class = ray.remote(cls)
                actor_handle = actor_class.options(**options_kwargs).remote(**cls_kwargs)
            except ValueError:
                pass


        
        return ray.get_actor(name)


def custom_getattr(obj, key):
    root_key = key.split('.')[0]
    rest_of_keys_path = '.'.join(key.split('.')[1:])

    if isinstance(obj, dict) and (root_key in obj):
        new_obj = obj[root_key]
    elif hasattr(obj, root_key):
        new_obj = getattr(obj, root_key)
    else:
        raise AttributeError(f'{root_key} not found')

    if len(rest_of_keys_path)>0:
        new_obj = custom_getattr(obj=new_obj,key=rest_of_keys_path)

    return new_ob
    


class RayEnv:
    supported_modes = ['second', 'timestamp']
    start_time = None
    


    def __init__(self, *args, **kwargs):


        self.ray_init_kwargs = kwargs
        self.ray_init_args = args
        self.ray_init_kwargs
        kwargs.update(locals())
        kwargs.pop('kwargs')
        self.__dict__.update(kwargs)
        self.ray_context = None
 

    @property
    def is_initialized(self):
        return bool(ray.is_initialized())


    def check_ray_init_kwargs(self):
        assert isinstance(kwargs, dict)
        kwargs =  self.ray_init_kwargs

        default_key2value = {
            'namespace': 'default'
        }
        default_key2type = {
            'address': str,
            'namespace': str
        }
        for k, t in default_key2type.items():
            kwargs[k] = kwargs.get(k, default_key2value.get(k))
            assert kwargs.get(k) != None, f"{k} should be not null {kwargs}"
            assert isinstance(kwargs[k], t), f"{k} should be {t}"
        
        self.ray_init_kwargs = kwargs


        return self.ray_init_kwargs






    @property
    def enter_context_gate(self):
        # if the context
        if self.ray_init_args == None or self.ray_init_args == False:
            return False
        elif len(self.ray_init_kwargs) == 0:
            return False
        else:
            self.check_ray_init_kwargs()
            return True

    def __enter__(self):
        if self.enter_context_gate:
            if self.is_initalized:
                raise Exception('The context is already initialized')
            else:
                self.ray_context = ray.init(**self.ray_init_args,**self.ray_init_kwargs)
        else:
            self.ray_context =  None


        return self.ray_context

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.ray_context != None :
            ray.shutdown()
        else:
            pass

