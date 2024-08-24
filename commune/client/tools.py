

import commune as c
import json

import commune as c
from functools import partial
import asyncio



class ClientTools:

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


  
    @classmethod
    def check_connection(cls, *args, **kwargs):
        return c.gather(cls.async_check_connection(*args, **kwargs))

    @classmethod
    def module2connection(cls,modules = None, network=None):
        if modules == None:
            modules = c.servers(network=network)
        connections = c.gather([ c.async_check_connection(m) for m in modules])

        module2connection = dict(zip(modules, connections))
    
        return module2connection

    @classmethod
    async def async_check_connection(cls, module, timeout=5, **kwargs):
        try:
            module = await cls.async_connect(module, return_future=False, virtual=False, **kwargs)
        except Exception as e:
            return False
        server_name =  await module(fn='server_name',  return_future=True)
        if cls.check_response(server_name):
            return True
        else:
            return False

    @staticmethod
    def check_response(x) -> bool:
        if isinstance(x, dict) and 'error' in x:
            return False
        else:
            return True
    
  
    def get_curl(self, 
                        fn='info', 
                        params=None, 
                        args=None,
                        kwargs=None,
                        timeout=10, 
                        module=None,
                        key=None,
                        headers={'Content-Type': 'application/json'},
                        network=None,
                        version=1,
                        mode='http',
                        **extra_kwargs):
            key = self.resolve_key(key)
            network = network or self.network
            url = self.get_url(fn=fn, mode=mode, network=network)
            kwargs = {**(kwargs or {}), **extra_kwargs}
            input_data = self.get_params(args=args, kwargs=kwargs, params=params, version=version)

            # Convert the headers to curl format
            headers_str = ' '.join([f'-H "{k}: {v}"' for k, v in headers.items()])

            # Convert the input data to JSON string
            data_str = json.dumps(input_data).replace('"', '\\"')

            # Construct the curl command
            curl_command = f'curl -X POST {headers_str} -d "{data_str}" "{url}"'

            return curl_command
    

    def run_curl(self, *args, **kwargs):
        curl_command = self.get_curl(*args, **kwargs)
        # get the output of the curl command
        import subprocess
        output = subprocess.check_output(curl_command, shell=True)
        return output.decode('utf-8')

