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
    endpoints = ['modules', 'add_module', 'remove',  'update', 'test',  'module', 'info', 'functions', 'n']
    modules_path = os.path.expanduser('~/.commune/api/modules')

    def __init__(self,
                expose_functions = ['chain/info', 'chain/forward', 'chain/stream', 'chain/stream_forward'],
                background:bool = False, 
                path='~/.commune/api', **kwargs):
        self.store = c.mod('store')(path)
        self.chain = c.mod('chain')()

    def paths(self):
        return self.ls(self.modules_path)

    def n(self, search=None):
        return len(self.names(search=search))

    def names(self, search=None, **kwargs):
        return  c.mods(search=search, **kwargs)

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
                    search=None,
                    page=1, 
                    update=False, 
                    modules:Optional[list]=None,
                     page_size=20, 
                    timeout=200, 
                    code=True,
                    df = False,
                    names = False,
                    threads=8,
                    features = ['name', 'schema', 'key'],
                    max_age=None, 
                    mode = 'process',
                    verbose=False, **kwargs):

        modules = self.names(search=search, update=update)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        modules = modules[start_idx:end_idx]
        progress_bar = c.tqdm(modules, desc=f"Loading modules thread={page}", total=len(modules))

        results = []
        if threads > 1:
            executor = self.executor(max_workers=threads, mode=mode)
            futures = []
            for module in modules:
                future = executor.submit(self.module, module, max_age=max_age, update=update, code=code)
                futures.append(future)
            for future in c.as_completed(futures):
                result = future.result()
                if isinstance(result, dict) and 'name' in result:
                    print(f"Module {result['name']} loaded")
                if self.check_module_data(result):
                    results.append(result)
                else:
                    c.print(result, color='red', verbose=verbose)
                progress_bar.update(1)
            executor.shutdown(wait=True)
        else:

            for module in modules:
                result = self.module(module, max_age=max_age, update=update)
                if self.check_module_data(result):
                    results.append(result)
                else: 
                    c.print(result, color='red', verbose=verbose)
                progress_bar.update(1)
        if df:
            results = c.df(results)
        if names:
            results = [m['name'] for m in results]
        return results

    def module(self, module:str, max_age=None, update=False, code=False, **kwargs):

        try:
            path = self.store.get_path(f'modules/{module}.json')
            info = c.get(path, None,  max_age=max_age, update=update)
            if info == None:
                info = c.info(module, max_age=max_age, update=update)
                module_path = self.module_path(module)
                info = load_json(module_path)["data"]
                info["code"] = c.code_map(info['name'])
                self.store.put(module, info)
            
        except Exception as e:

            print(f"Error({module} error:{e})")
        if not code:
            info.pop("code", None)
        return info


    allowed_functions = ['chain/events', 'chain/forward', 'chain/stream', 'chain/stream_forward', 'schema']

    def call(self, fn, params={"text": "hey"}, **kwargs):
        assert fn in self.allowed_functions, f"Function {fn} is not allowed to be called directly. Use one of the allowed functions: {self.allowed_functions}"
        return c.fn(fn)(**params, **kwargs)

    def ask(self, text, **kwargs):
        return c.ask(text, **kwargs)
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
                   code = None, 
                   key  = None, 
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


    def sync(self, max_age=10000, page_size=32, threads=8):

        n = self.n()
        pages = n // page_size + 1
        print(f"Syncing {n} modules in {pages} pages with page size {page_size}")
        for page in range(1, pages + 1):
            print(f"Syncing page {page}/{pages}")
            self.modules(search=None, max_age=max_age, page=page, page_size=page_size, threads=threads)
        return {"message": f"Synced {n} modules in {pages} pages with page size {page_size}"}


    def balance(self, address):
           
        return self.chain.balance(address)

