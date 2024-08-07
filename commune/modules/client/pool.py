

import commune as c
class ClientPool:

    @classmethod
    def call_search(cls, 
                    search : str, 
                *args,
                timeout : int = 10,
                network:str = 'local',
                key:str = None,
                kwargs = None,
                **extra_kwargs) -> None:
        if '/' in search:
            search, fn = search.split('/')
        namespace = c.namespace(search=search, network=network)
        future2module = {}
        for module, address in namespace.items():
            c.print(f"Calling {module}/{fn}", color='green')
            future = c.submit(cls.call,
                               args = [module, fn] + list(args),
                               kwargs = {'timeout': timeout, 
                                         'network': network, 'key': key, 
                                         'kwargs': kwargs,
                                         **extra_kwargs} , timeout=timeout)
            future2module[future] = module
        futures = list(future2module.keys())
        result = {}
        progress_bar = c.tqdm(len(futures))
        for future in c.as_completed(futures, timeout=timeout):
            module = future2module.pop(future)
            futures.remove(future)
            progress_bar.update(1)
            result[module] = future.result()

        return result
            
    
    @classmethod
    def call_pool(cls, 
                    modules, 
                    fn = 'info',
                    *args, 
                    network =  'local',
                    timeout = 10,
                    n=None,
                    **kwargs):
        
        args = args or []
        kwargs = kwargs or {}
        
        if isinstance(modules, str) or modules == None:
            modules = c.servers(modules, network=network)
        if n == None:
            n = len(modules)
        modules = cls.shuffle(modules)[:n]
        assert isinstance(modules, list), 'modules must be a list'
        futures = []
        for m in modules:
            job_kwargs = {'module':  m, 'fn': fn, 'network': network, **kwargs}
            future = c.submit(c.call, kwargs=job_kwargs, args=[*args] , timeout=timeout)
            futures.append(future)
        responses = c.wait(futures, timeout=timeout)
        return responses
    

    @classmethod
    def connect_pool(cls, modules=None, *args, return_dict:bool=False, **kwargs):
        if modules == None:
            modules = c.servers(modules)
        
        module_clients =  cls.gather([cls.async_connect(m, ignore_error=True,**kwargs) for m in modules])
        if return_dict:
            return dict(zip(modules, module_clients))
        return module_clients


