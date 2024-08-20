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
           cmd = None,
           update:bool=False,
           cwd = None, 
           **extra_kwargs):
        

        module = c.shortcuts().get(module, module)
        app2info = self.get('app2info', {})
        kwargs = kwargs or {}
        name = name or module
        port = port or app2info.get(name, {}).get('port', c.free_port())
        if update:
            port = c.free_port()
        process_name = self.name_prefix + name 
        if c.port_used(port):
            c.kill_port(port)
            c.pm2_kill(process_name)
        if c.module_exists(module + '.app'):
            module = module + '.app'
        kwargs_str = json.dumps(kwargs or {}).replace('"', "'")
        module_class = c.module(module)
        cmd = cmd or f'streamlit run {module_class.filepath()} --server.port {port} -- --fn {fn} --kwargs "{kwargs_str}"'


        cwd = cwd or os.path.dirname(module_class.filepath())

        if remote:
            rkwargs = c.locals2kwargs(locals())
            rkwargs['remote'] = False
            del rkwargs['module_class']
            del rkwargs['app2info']
            del rkwargs['process_name']
            self.remote_fn(
                        fn='start',
                        name=self.name_prefix + name ,
                        kwargs= rkwargs)
        
            return {
                'name': name,
                'cwd': cwd,
                'fn': fn,
                'address': {
                    'local': f'http://localhost:{port}',
                    'public': f'http://{c.ip()}:{port}',
                }  

            }

        module = c.module(module)
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
        c.cmd(cmd, verbose=True, cwd=c.pwd())
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
    
    def app_modules(self, **kwargs):
        return list(set([m.replace('.app','') for m in self.modules() if self.has_app(m, **kwargs)]))
    
    def is_running(self, name):
        return self.resolve_process_name(name) in self.apps()
    
    def resolve_process_name(self, name):
        return self.name_prefix + name

    def run_all(self):
        for app in self.app_modules():
            c.print(self.start(app))

    def app2url(self):
        app2info = self.app2info()
        app2url = {}
        for app, info in app2info.items():
            port = info['port']
            if c.port_used(port):
                app2url[app] = '0.0.0.0:' + str(info['port'])
        return app2url

    # def app(self, **kwargs):
    #     # create a dashbaord of all apps
    #     app_modules = self.app_modules(**kwargs)


        
App.run(__name__)
     