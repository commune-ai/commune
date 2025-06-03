from fastapi import FastAPI, HTTPException
import uvicorn
import os
import json
from pydantic import BaseModel
from typing import Dict, Optional
import commune as c 
# Pydantic model for module dat
import requests
import requests
from .utils import load_json, save_json, logs

class Api:

    tempo = 600
    app_name =  __file__.split('/')[-3] + '_app' 
    model='anthropic/claude-3.5-sonnet'
    endpoints = [
                'modules', 
                'add_module', 
                'remove', 
                'update', 
                'test', 
                'get_module',
                'info', 
                'functions']

    modules_path = os.path.expanduser('~/.commune/api/modules')

    def __init__(self, background:bool = False, path='~/.commune/api', **kwargs):

        self.store = c.module('store')(path)
        if background:
            print(c.serve('api::background'))

    def __delete__(self):
        c.kill('api::background')
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
        return len(c.get_modules())

    def names(self, search=None):
        return  c.get_modules(search=search)

    def modules(self, 
                    names:Optional[list]=None,
                        max_age=None, 
                        update=False, 
                        lite=False, 
                        search=None,
                        page=1, 
                        timeout=60, 
                        page_size=100, 
                        df = False,
                        threads=1,
                        features = ['name', 'schema', 'key'],
                        mode = 'default',
                        verbose=False):


        names = names or self.names()
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        names = names[start_idx:end_idx]

        if search != None:
            names = [n for n in names if search in n]
        if threads > 1:
            params = locals()
            params.pop('self')
            params['threads'] = 1
            n = len(names) 
            params_list = []
            params['page_size']  = n // threads
            for i in range(1, threads+1):
                if i == threads:
                    page_size = n - (threads-1) * page_size
                else:
                    page_size = n // threads
                params['page'] = i
                params_list.append(params)

            futures = []
            results = []
            for params in params_list:
                c.print(params)
                future = c.submit(self.modules, params, timeout=timeout)
                futures.append(future)
            results = []
            try:
                for future in c.as_completed(futures, timeout=timeout):
                    result = future.result()
                    print(result)
                    results.extend(result)
            except TimeoutError as e:
                print(f"TimeoutError: {e}")
            return results
        elif threads == 1:
            results = []
            futures = []
            future2module = {}
            progress_bar = c.tqdm(names, desc=f"Loading modules thread={page}", total=len(names))
            fails = 0
            for module_name in names:
                path = self.store.get_path(f'modules/{module_name}.json')
                result = c.get(path, None,  max_age=max_age, update=update)
                if result == None:
                    try:
                        result = c.info(module_name, max_age=max_age, update=update)
                        c.put(path, result)
                    except Exception as e:
                        result = c.detailed_error(e)
                        fails += 1
                progress_bar.update(1)
                results.append(result)
        else:
            raise Exception(f'thread number not supported thread>=1 vs {thread}')

        # results =  list(filter(result_filter, results))
        if mode == 'n':
            results = len(results)
        if df:
            results = c.df(results)
        return results

    def get_module(self, module:str, **kwargs):
        if not self.module_exists(module):
            raise HTTPException(status_code=404, detail="Module not found")
        module_path = self.get_module_path(module)
        info = load_json(module_path)["data"]
        
        return info

    def get_module_path(self, module):
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
            module = self.get_module(module)
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
        module_path = self.get_module_path(module["name"])
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
        os.remove(self.get_module_path(module))
        return {"message": f"Module {module} removed successfully"}

    def module_exists(self, module: str):
        return os.path.exists(self.get_module_path(module))

    def update(self, module: str):
        if not self.module_exists(module):
            raise HTTPException(status_code=404, detail="Module not found")
        module = self.get_module(module)
        self.save_module(module)