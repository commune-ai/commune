import commune as c
import os
import json
import streamlit as st
from typing import *

class App(c.Module):
    port_range = [8501, 8600]
    name_prefix = 'app::'





    def start(self,
           module:str = 'server', 
           name : Optional[str] = None,
           fn:str='app', 
           port:int=None, 
           remote:bool = True, 
           kwargs:dict=None, 
           update:bool=False,
           cwd = None, 
           **extra_kwargs):

        module = c.shortcuts().get(module, module)
        app2info = self.get('app2info', {})
        kwargs = kwargs or {}
        name = name or module
        port = port or c.free_port()
        if c.port_used(port):
            c.kill_port(port)
        if c.module_exists(module + '.app'):
            module = module + '.app'

        if remote:
            kwargs = c.locals2kwargs(locals())
            self.remote_fn(
                        fn='start',
                        name=self.name_prefix + name ,
                        kwargs= kwargs)
        
            return {
                'name': name,
                'fn': fn,
                'address': {
                    'local': f'http://localhost:{port}',
                    'public': f'http://{c.ip()}:{port}',
                },

            }

        module = c.module(module)
        kwargs_str = json.dumps(kwargs or {}).replace('"', "'")
        cmd = f'streamlit run {module.filepath()} --server.port {port} -- --fn {fn} --kwargs "{kwargs_str}"'
        cwd = cwd or os.path.dirname(module.filepath())
        app_info= {
            'name': name,
            'port': port,
            'fn': fn,
            'kwargs': kwargs,
            'cmd': cmd, 
            'cwd': cwd ,
        }
        app2info[name] = app_info
        self.put('app2info', app2info )
        
        c.cmd(cmd, verbose=True, cwd=cwd)
        return app_info
    
    start_app = app = start


    def app2info(self):
        app2info =  self.get('app2info', {})
        if not isinstance(app2info, dict):
            app2info = {}
        return app2info
    

    def kill_all(self):
        return c.module('pm2').kill_many(self.apps())
    
    def kill(self, name):
        return c.module('pm2').kill(self.name_prefix+name)
    
    def filter_name(self, name:str) -> bool:
        return bool(name.startswith(self.name_prefix))
        
    def apps(self, remove_prefix = True):
        apps =  [n for n in c.pm2ls() if n.startswith(self.name_prefix)]
        if remove_prefix:
            apps = [n[len(self.name_prefix):] for n in apps]
        return apps
    
    
    


    
    

App.run(__name__)
    

c