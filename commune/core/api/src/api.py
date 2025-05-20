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
    free = True
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


    def module_names(self):
        return  c.get_modules()

    
    def modules(self, 
                        max_age=None, 
                        update=False, 
                        lite=False, 
                        search=None,
                        page=1, 
                        timeout=60, 
                        page_size=10, 
                        threads=1,
                        mode = 'n',
                        verbose=False):

        if threads > 1:
            n = self.n()
            page_size = n // threads
            og_params = {'max_age': max_age, 'update': update, 'lite': lite, 'search': search, 'page': page, 'timeout': timeout, 'page_size': page_size, 'threads': 1}
            params_list = [ {**og_params , **{'page': i, 'page_size': page_size}} for i in range(1, threads+1)]
            futures = []
            results = []
            for params in params_list:
                future = c.submit(self.modules, params, timeout=timeout)
                futures.append(future)
            results = []
            try:
                for future in c.as_completed(futures, timeout=timeout):
                    result = future.result()
                    if result is not None:
                        results.extend(result)
            except TimeoutError as e:
                print(f"TimeoutError: {e}")
            return results

        module_names = self.module_names()
        if search != None:
            module_names = [n for n in module_names if search in n]

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        module_names = module_names[start_idx:end_idx]

        results = []
        futures = []
        future2module = {}

        progress_bar = c.tqdm(module_names, desc=f"Loading modules thread={page}", total=len(module_names))
        fails = 0
        for module_name in module_names:
            path = self.store.get_path(f'modules/{module_name}.json')
            result = c.get(path, None,  max_age=max_age, update=update)
            if result == None:
                try:
                    result = c.info(module_name, max_age=max_age, update=update)
                except Exception as e:
                    result = c.detailed_error(e)
                    fails += 1
                finally:
                    c.put(path, result)
                    progress_bar.update(1)
            results.append(result)
        result_filter = lambda x: bool(isinstance(x, dict) and 'name' in x and 'schema' in x and 'key' in x)
        results =  list(filter(result_filter, results))
        if mode == 'n':
            results = sorted(results, key=lambda x: x['name'])
        return results

    def check_info(self, info, features=['name', 'schema', 'key']):
        return isinstance(info, dict) and all([f in info for f in features])

    def names(self):
        return [m['name'] for m in self.modules()]

    def get_module(self, module:str, **kwargs):
        info =  c.info(module, lite=False, **kwargs)
        prefix = info['name'].split('.')[0]
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

    def clear_modules(self):
        for module_path in self.ls(self.modules_path):
            print('Removing:', module_path)
            os.remove(module_path)
        return {"message": "All modules removed"}
    
    def resolve_path(self, path):
        return os.path.expanduser('~/.hub/api/') + path

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

    def root():
        return {"message": "Module Management API"}


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

    def test(self):
        
        # Test module data
        test_module = {
            "name": "test_module",
            "url": "http://test.com",
            "key": "test_key",
            "key_type": "string",
            "description": "Test module description"
        }
        # Add module
        self.add_module(test_module)
        assert self.module_exists(test_module['name']), "Module not added"
        self.remove_module(test_module['name'])
        assert not self.module_exists(test_module['name']), "Module not removed"
        return {"message": "All tests passed"}
    

    def get_key(self, password, **kwargs):
        return c.str2key(password, **kwargs)
