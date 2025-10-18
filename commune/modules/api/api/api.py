from fastapi import  HTTPException
import uvicorn
import os
import json
from typing import Dict, Optional
import requests
from .utils import load_json, save_json, logs
import commune as c 

class Api:

    endpoints = ['modules', 'add_module', 'remove',  'update', 'test',  'module', 'info', 'functions', 'n', 'ask', 'mods', 'mod']
    port = 8000
    url = '0.0.0.0:8000'
    tempo = 600
    mods_path = os.path.expanduser('~/.commune/api/modules')

    def __init__(self,
                expose_functions = ['chain/info', 'chain/forward', 'chain/stream', 'chain/stream_forward'],
                background:bool = False, 
                path='~/.commune/api', **kwargs):
        self.store = c.mod('store')(path)
        print(c.mod('chain'))
        self.chain = c.mod('chain')()

    def paths(self):
        return self.ls(self.mods_path)

    def n(self, search=None):
        return len(self.names(search=search))

    def names(self, search=None, **kwargs):
        return  [m.split('.')[0] for m in c.mods(search=search, **kwargs)]

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

    def mods(self, 
                    search=None,
                    page=1, 
                    update=False, 
                    content=False,
                    modules:Optional[list]=None,
                    page_size=20, 
                    timeout=200, 
                    schema = True,
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
                future = executor.submit(
                                        self.module,
                                        module, 
                                        max_age=max_age, 
                                        update=update, 
                                        content=content, 
                                        schema=schema
                                        )
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
                print(f"Loading module {module}")
                result = self.mod(module, update=update, content=content, schema=schema)
                if self.check_module_data(result):
                    results.append(result)
                else: 
                    c.print(result, color='red', verbose=verbose)
                progress_bar.update(1)
        if df:
            results = c.df(results)
        if names:
            results = [m['name'] for m in results if 'name' in m]
        return results

    modules = mods

    def mod(self, module:str,  update=False,  content=False, schema = False, public= False, **kwargs):
        """
        Get module info
        1. Check if module info is in store and not expired
        2. If not, fetch module info from module server
        """
        module = module.replace('.', '/')
        try:
            path = f'modules/{module}.json'
            info = self.store.get(path, None, update=update)
            if info == None:
                # fetch module info from module server
                print(f"Fetching module {module} info from server -> {path}...")
                info = c.info(module=module, content=True, schema=True, public=True, **kwargs)
                self.store.put(path, info)
            if not content:
                info.pop("content", None)
                info.pop('code', None)
            if not schema:
                info.pop("schema", None)
            if not public: 
                # remove all of the content from schema and info
                info.pop("content", None)
                for k in list(info.get("schema", {}).keys()):
                    info["schema"][k].pop("content", None)
        except Exception as e:
            c.print(module,c.detailed_error(e), color='red')
        return info

    module = mod

    def servers(self):
        return c.servers()

    def call(self, fn, 
                params={}, 
                fns = ['chain/events', 
                         'chain/forward', 
                         'chain/stream', 
                         'chain/stream_forward', 
                         'schema'],
                auth = None, **kwargs):

        assert fn in fns, f"Function {fn} is not allowed to be called directly. Use one of the allowed functions: {self.allowed_functions}"
        return c.fn(fn)(**params, **kwargs)

    def ask(self, text, **kwargs):
        return c.ask(text, **kwargs)

    def check_module_data(self, module) -> bool:
        if not isinstance(module, dict):
            return False
        features = ['name', 'key', 'schema']
        return True

    def module_path(self, module):
        return f"{self.mods_path}/{module}.json"

    def ls(self, path=mods_path):
        if not os.path.exists(path):
            print('WARNING IN LS --> Path does not exist:', path)
            return []
        path = os.path.abspath(path)
        return c.ls(path)

    def check_module(self, module):
        features = ['name', 'url', 'key']  
        if isinstance(module, str):
            module = self.mod(module)
        if not isinstance(module, dict):
            return False
        assert all([f in module for f in features]), f"Missing feature in module: {module}"
        return True

    def save_module(self, module):
        print('SAVING MODULE', module["name"])
        module_path = self.module_path(module["name"])
        save_json(module_path, module)
        return {"message": f"Module {module['key']} updated successfully"}

    def info(self, module:str, **kwargs):
        return c.info(module,  **kwargs)

    def add_module(self, 
                   name  = "module", 
                   source = None, 
                   key  = None, 
                   url  = "0.0.0.0:8000", 
                   app = None,
                   **kwargs ):
        
        module = { "name": name, "url": url, "key": key, "source": source,  **kwargs }
        self.save_module(module)
        result =  {"message": f"Module {module['name']} added successfully", "module": module}
        print('RESULT',result)
        return result

    def remove(self, module: str):
        assert self.mod_exists(module), "Module not found"
        os.remove(self.module_path(module))
        return {"message": f"Module {module} removed successfully"}

    def mod_exists(self, module: str):
        return os.path.exists(self.module_path(module))

    def sync(self, max_age=10000, page_size=32, threads=8):

        n = self.n()
        pages = n // page_size + 1
        print(f"Syncing {n} modules in {pages} pages with page size {page_size}")
        for page in range(1, pages + 1):
            print(f"Syncing page {page}/{pages}")
            self.mods(search=None, max_age=max_age, page=page, page_size=page_size, threads=threads)
        return {"message": f"Synced {n} modules in {pages} pages with page size {page_size}"}

    def balance(self, address):
        return self.chain.balance(address)

    def add_host(self, host='0.0.0.0', port=8000, **kwargs):
        """
        Add a host to the commune
        """
        ip = f"{host}:{port}"
        path = 'api/hosts.json'
        hosts = self.store.get(path, [self.url])
        hosts = list(set(hosts))
        assert isinstance(hosts, list), "Hosts should be a list"
        assert ip not in hosts, f"Host {ip} already exists in the list"
        self.store.put('api/hosts', path)
        return {"message": f"Host {host}:{port} added successfully"}

    def remove_host(self, host='0.0.0.0', port=8000, **kwargs):
        """
        Remove a host from the commune
        """
        ip = f"{host}:{port}"
        path = 'api/hosts.json'
        hosts = self.store.get(path, [self.url])
        if ip in hosts:
            hosts.remove(ip)
            self.store.put('api/hosts', path)
            return {"message": f"Host {host}:{port} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Host {ip} not found in the list")

    def hosts(self, update=False):
        """
        Get the list of hosts
        """
        path = 'api/hosts.json'
        hosts = self.store.get(path, [self.url], update=update)
        return hosts

    def clear_hosts(self):
        """
        Clear the list of hosts
        """
        path = 'api/hosts.json'
        self.store.put('api/hosts', path, [])
        return {"message": "Hosts cleared successfully"}

    def ask(self, text, *extra_text,  **kwargs):
        """
        Ask a question to the commune
        """
        return c.ask(text, *extra_text, **kwargs)


