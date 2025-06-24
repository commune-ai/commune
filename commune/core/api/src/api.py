from fastapi import  HTTPException
import uvicorn
import os
import json
from typing import Dict, Optional
import requests
from .utils import load_json, save_json, logs
import commune as c 

class Api:

    tempo = 600
    app_name =  __file__.split('/')[-3] + '_app' 
    model='anthropic/claude-3.5-sonnet'
    endpoints = ['modules', 'add_module', 'remove',  'update', 'test',  'module', 'info', 'functions']
    modules_path = os.path.expanduser('~/.commune/api/modules')

    def __init__(self, background:bool = False, path='~/.commune/api', **kwargs):
        self.store = c.mod('store')(path)
        if background:
            print(c.serve('api:background'))

    def __delete__(self):
        c.kill('api:background')
        return {"message": "Background process killed"}

    def background_loop(self, sleep_initial=10, max_age=100, threads=2):
        print('Starting background loop')
        step = 0
        while True:
            step += 1
            c.sleep(max_age/2)

            print('Background loop step:', step)
            
            self.modules(max_age=max_age, threads=threads)
            print('Background loop step:', step, 'completed')

    def paths(self):
        return self.ls(self.modules_path)

    def n(self):
        return len(c.mods())

    def names(self, search=None):
        return  c.mods(search=search)

    def executor(self,  max_workers=8, mode='thread'):
        if mode == 'process':
            from concurrent.futures import ProcessPoolExecutor
            executor =  ProcessPoolExecutor(max_workers=max_workers)
        elif mode == 'thread':
            from concurrent.futures import ThreadPoolExecutor
            executor =  ThreadPoolExecutor(max_workers=max_workers)
        elif mode == 'async':
            from commune.core.api.src.async_executor import AsyncExecutor
            executor = AsyncExecutor(max_workers=max_workers)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'thread', 'process' or 'async'.")
        return executor

    def modules(self, 
                    modules:Optional[list]=None,
                    update=False, 
                    search=None,
                    page=1, 
                     page_size=100, 
                    timeout=200, 
                    code=True,
                    df = False,
                    threads=16,
                    features = ['name', 'schema', 'key'],
                    max_age=None, 
                    mode = 'process',
                    verbose=False, **kwargs):

        if update:
            mode= 'process'
        else:
            mode = 'thread'

        if modules == None:
            modules = self.names()
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            modules = modules[start_idx:end_idx]

        if search != None:
            modules = [m for m in modules if search in m]
        
        if len(modules) == 0:
            print('No modules found')
            return []

        if threads > 1:
            params = locals()
            rm_features = ['self', 'modules', 'verbose', 'start_idx', 'end_idx']
            for f in rm_features:
                params.pop(f, None)
            params['threads'] = 1
            results = []
            futures = []
            batch_size = len(modules) // threads
            if batch_size == 0:
                batch_size = 1
            print(f"Loading {len(modules)} modules in batches of {batch_size} with {threads} threads")
            modules_chunks = [ modules[i:i + batch_size] for i in range(0, len(modules), batch_size) ]

            # from concurrent.futures import ThreadPoolExecutor
            futures = []
            executor = self.executor(max_workers=threads, mode=mode)
            for i, modules_chunk in enumerate(modules_chunks):
                params['modules'] = modules_chunk
                futures.append(executor.submit(self.modules, **params))
            for future in c.as_completed(futures, timeout=timeout):
                try:
                    results.extend(future.result())
                except Exception as e:
                    if verbose:
                        c.print(f"Error in future: {e}", color='red')
            executor.shutdown(wait=True)
        else:
            # if update and max_age == None:
            #     max_age = 600
            #     update = False
            progress_bar = c.tqdm(modules, desc=f"Loading modules thread={page}", total=len(modules))
            results = []
            for module in modules:
                result = self.module(module, max_age=max_age, update=update)
                if self.check_module_data(result):
                    results.append(result)
                else: 
                    c.print(result, color='red', verbose=verbose)
                progress_bar.update(1)
        if df:
            results = c.df(results)
        return results

    def module(self, module:str, max_age=None, update=False, code=True):

        path = self.store.get_path(f'modules/{module}.json')
        result = c.get(path, None,  max_age=max_age, update=update)
        if result == None:
            try:
                result = c.info(module, max_age=max_age, update=update ,code=True)
            except Exception as e:
                result = c.detailed_error(e)
            c.put(path, result)
        module_path = self.module_path(module)
        if not code:
            result.pop('code', None)
        info = load_json(module_path)["data"]
        return info

    def check_module_data(self, module) -> bool:
        if not isinstance(module, dict):
            return False
        features = ['name', 'key', 'schema']
        return all([f in module for f in features])

    def module_path(self, module):
        return f"{self.modules_path}/{module}.json"

    def ls(self, path=modules_path):
        if not os.path.exists(path):
            print('WARNING IN LS --> Path does not exist:', path)
            return []
        path = os.path.abspath(path)
        return c.ls(path)

    def logs(name):
        return c.logs(name)

    def check_module(self, module):
        features = ['name', 'url', 'key']  
        if isinstance(module, str):
            module = self.module(module)
        if not isinstance(module, dict):
            return False
        assert all([f in module for f in features]), f"Missing feature in module: {module}"
        return True

    def check_modules(self):
        checks = []
        for m in self.infos():
            try:
                self.check_module(m)
                m['check'] = True
            except Exception as e:
                print(e)
                m['check'] = False
            checks += [m]
        return checks

    def save_module(self, module):
        print('SAVING MODULE', module["name"])
        module_path = self.module_path(module["name"])
        save_json(module_path, module)
        return {"message": f"Module {module['key']} updated successfully"}

    def info(self, module:str, **kwargs):
        return c.info(module,  **kwargs)

    def add_module(self, 
                   name  = "module", 
                   key  = "module_key", 
                   code = None, 
                   url  = "0.0.0.0:8000", 
                   app = None,
                   **kwargs ):
        
        module = { "name": name, "url": url, "key": key, "code": code,  **kwargs }
        self.save_module(module)
        result =  {"message": f"Module {module['name']} added successfully", "module": module}
        print('RESULT',result)
        return result

    def remove(self, module: str):
        assert self.module_exists(module), "Module not found"
        os.remove(self.module_path(module))
        return {"message": f"Module {module} removed successfully"}

    def module_exists(self, module: str):
        return os.path.exists(self.module_path(module))
