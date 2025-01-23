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
class Backend:
    
    server_port = 8000
    app_port = 3000
    app_name =  __file__.split('/')[-3] + '_app' 
    model='anthropic/claude-3.5-sonnet'
    free = True
    endpoints = ["get_modules", 'modules', 'add_module', 'remove', 'update', 'test', 'info']
    modules_path = __file__.replace(__file__.split('/')[-1], 'modules')

    # In-memory storage for modules
    
    def get_module_path(self, module_id):
        return f"{self.modules_path}/{module_id}.json"

    def ls(self, path=modules_path):
        if not os.path.exists(path):
            print('WARNING IN LS --> Path does not exist:', path)
            return []
        path = os.path.abspath(path)
        return c.ls(path)

    def logs(name):
        return c.logs(name)

    def check_module(self, module):
        features = ['name', 'address', 'key', 'code']  
        if isinstance(module, str):
            module = self.get_module(module)
        if not isinstance(module, dict):
            return False
        assert all([k not in module for k in features])
        return True

    def check_modules(self):
        checks = []
        for m in self.modules():
            try:
                self.check_module(m)
                m['check'] = True
            except Exception as e:
                print(e)
                m['check'] = False
            checks += [m]
        return checks

    def save_module(self, module):
        self.check_module(module)
        module_path = self.get_module_path(module["key"])
        save_json(module_path, module)
        return {"message": f"Module {module['key']} updated successfully"}

    def get_module(self, module_id):
        module_path = self.get_module_path(module_id)
        return load_json(module_path)
    
    def clear_modules(self):
        for module_path in self.ls(self.modules_path):
            print('Removing:', module_path)
            os.remove(module_path)
        return {"message": "All modules removed"}
    
    def get_modules(self):
        return self.modules()
    
    def resolve_path(self, path):
        return os.path.expanduser('~/.hub/api/' + path)
    
    def module_paths(self):
        return self.ls(self.resolve_path('modules'))

    def modules(self, max_age=600, search=None, update=False, lite=True, page=1, page_size=100):
        module_paths = self.module_paths()
        module_infos = []
        progress = c.progress(len(module_paths))
        for module_path in module_paths:
            module_info = c.get(module_path, lite=lite, max_age=max_age, update=update)
            module_name = module_path.split('/')[-1].split('.json')[0]
            if search is not None and search not in module_name:
                continue
            if module_info is None:
                try:
                    code = c.code(module_name)
                    hash_code = c.hash(code)
                    key = c.pwd2key(hash_code)
                    module_info = {
                        'name': module_name, 
                        'code': code,
                        'hash': hash_code, 
                        'size': len(code),
                        'time': c.time(),
                        'key': key.ss58_address, 
                        }
                    if lite:
                        module_info.pop('code')
                    c.put(module_path, module_info)
                except Exception as e:
                    print('ERROR', e)
                    continue

    
            progress.update(1)
            
            module_infos += [module_info]
            
        return module_infos

    def add_module(self, name  = "module", 
                   key  = "module_key", 
                   code = None, 
                   address  = "0.0.0.0:8000", 
                   **kwargs ):
        
        module = { "name": name, "address": address, "key": key, "code": code, }
        self.save_module(module)
        result =  {"message": f"Module {module['name']} added successfully", "module": module}
        print('RESULT',result)
        return result

    def root():
        return {"message": "Module Management API"}

    def get_module(self, module_id: str):
        modules = self.get_modules()
        if module_id not in modules:
            raise HTTPException(status_code=404, detail="Module not found")
        return modules[module_id]

    def remove(self, module_id: str):
        assert self.module_exists(module_id), "Module not found"
        os.remove(self.get_module_path(module_id))
        return {"message": f"Module {module_id} removed successfully"}

    def module_exists(self, module_id: str):
        return os.path.exists(self.get_module_path(module_id))

    def get_modules(self):
        return load_json(self.modules_path)

    def update(self, module_id: str, module: Dict):
        if not self.module_exists(module_id):
            raise HTTPException(status_code=404, detail="Module not found")
        module = self.get_module(module_id)
        
        self.save_module(module_id, module)

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
    


    avoid_terms = ['__pycache__', '.ipynb_checkpoints', "node_modules", ".git", ".lock", "public", "json"]

    def serve(self, port=server_port):
        return c.serve(self.module_name(), port=port)
    
    def kill_app(self, name=app_name, port=app_port):
        while c.port_used(port):
            c.kill_port(port)
        return c.kill(name)

    
    def app(self, name=app_name, port=app_port, server_port=server_port, remote=0):
        self.serve(server_port)
        self.kill_app(name, port)
        c.cmd(f"pm2 start yarn --name {name} -- dev --port {port}")
        return c.logs(name, mode='local' if remote else 'cmd')
    
    def fix(self, name=app_name, model=model):
        logs = c.logs(name, mode='local')
        files =   self.files(f"{logs}")
        context = {f: c.get_text(f) for f in files}
        prompt = f"CONTEXT {context} LOGS  {logs} OBJECTIVE fix the issue"
        print('Sending prompt:',len(prompt))
        return c.ask(prompt[:10000], model=model)

    def query(self,  
              options : list,  
              query='most relevant files', 
              output_format="list[[key:str, score:float]]",  
              anchor = 'OUTPUT', 
              threshold=0.5,
              n=10,  
              model=model):

        front_anchor = f"<{anchor}>"
        back_anchor = f"</{anchor}>"
        output_format = f"DICT(data:{output_format})"
        print(f"Querying {query} with options {options}")
        prompt = f"""
        QUERY
        {query}
        OPTIONS 
        {options} 
        INSTRUCTION 
        get the top {n} functions that match the query
        OUTPUT
        (JSON ONLY AND ONLY RESPOND WITH THE FOLLOWING INCLUDING THE ANCHORS SO WE CAN PARSE) 
        {front_anchor}{output_format}{back_anchor}
        """
        output = ''
        for ch in c.ask(prompt, model=model): 
            print(ch, end='')
            output += ch
            if ch == front_anchor:
                break
        if '```json' in output:
            output = output.split('```json')[1].split('```')[0]
        elif front_anchor in output:
            output = output.split(front_anchor)[1].split(back_anchor)[0]
        else:
            output = output
        output = json.loads(output)
        assert len(output) > 0
        return [k for k,v in output["data"] if v > threshold]

    def files(self, query='the file that is the core of commune',  path='./',  n=10, model='anthropic/claude-3.5-sonnet-20240620:beta'):
        files =  self.query(options=c.files(path), query=query, n=n, model=model)
        return [c.abspath(path+k) for k in files]
    
    networks = ['ethereum',
                 'bitcoin', 
                 'solana', 
                 'bittensor', 
                 'commune']
    def is_valid_network(self, network):
        return network in self.networks
    
    def get_key(self, password, **kwargs):
        return c.str2key(password, **kwargs)

    def feedback(self, path='./',  model=model):
        code = c.file2text(path)
   
        prompt = f"""

        PROVIDE FEEDBACK and s a score out of 100 for the following prompt on quality 
        and honesty. I want to make sure these are legit and there is a good chance 
        they are not. You are my gaurdian angel.
        {code}        
        OUTPUT_FORMAT MAKE SURE ITS IN THE LINES
        <OUTPUT><DICT(pointers:str, score:int (out of 100))></OUTPUT>
        OUTPUT
        """

        return c.ask(prompt, model=model)


    def file2text(owner: str, repo: str, filepath: str, branch: str = 'main') -> str:
        """
        Get the text contents of a file in a GitHub repository without using the GitHub API.
        This uses the raw.githubusercontent.com domain to fetch the file content directly.
        
        Parameters:
            owner (str): Repository owner/organization
            repo (str): Repository name
            filepath (str): Path to the file within the repository
            branch (str): The branch to read from (default: 'main')
            
        Returns:
            str: The text content of the file.
        """
        raw_url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filepath}'
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text

    # Example Usage:
    # files = list_github_repo_files('commune-ai', 'commune')
    # for f in files:
    #     print(f['name'], f['path'], f['type'])
    #     if f['type'] == 'file':
    #         content = file2text('commune-ai', 'commune', f['path'])
    #         print(content)
