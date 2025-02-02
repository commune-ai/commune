import commune as c
import os
import json
import streamlit as st
from typing import *

class App(c.Module):
    name_prefix = 'app::'

    def get_free_port(self, module, port=None, update=False):
        app2info = self.get('app2info', {})
        if update:
            return c.free_port()
        port = app2info.get(module, {}).get('port', None)
        if port == None:
            port = c.free_port()
        return port

    def start(self,
           module:str = 'server', 
           name : Optional[str] = None,
           port:int=None, 
           remote:bool = False, 
           kwargs:dict=None, 
           cmd = None,
           update:bool=False,
           process_name:str=None,
           cwd = None):
        port = self.get_free_port(module=module, port=port, update=update)
        
        rkwargs = c.locals2kwargs(locals())
        rkwargs['remote'] = False
        response = {'success': True,'module': module,'url':  f'http://localhost:{port}'}
        module = c.shortcuts.get(module, module)
        kwargs = kwargs or {}
        name = name or module
        port = port or self.get_free_port(module)
        # if the process is already running, kill it
        # if the module is an app, we need to add the .app to the module name
        if c.module_exists(module + '.app'):
            module = module + '.app'
        module_class = c.module(module)
        cmd = cmd or f'streamlit run {module_class.filepath()} --server.port {port}'
        return c.cmd(cmd)

    start_app = app = start


    def app2info(self):
        app2info =  self.get('app2info', {})
        if not isinstance(app2info, dict):
            app2info = {}
        changed = False
        og_app2info = app2info.copy()
        for name, info in og_app2info.items():
            if not c.port_used(info['port']):
                c.print(f'Port {info["port"]} is not used. Killing {name}')
                changed = True
                del app2info[name]
        if changed:
            self.put('app2info', app2info)

        return app2info

    def kill_all(self):
        return c.module('server').kill_many(self.apps())
    
    def kill_app(self, name):
        return c.module('server').kill(self.name_prefix+name)
    
    def filter_name(self, name:str) -> bool:
        return bool(name.startswith(self.name_prefix))
        
    def apps(self, remove_prefix = True):
        apps =  [n for n in c.pids() if n.startswith(self.name_prefix)]
        if remove_prefix:
            apps = [n[len(self.name_prefix):] for n in apps]
        return apps
    

    def app_exists(self, name):
        return name in self.apps()
    

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
