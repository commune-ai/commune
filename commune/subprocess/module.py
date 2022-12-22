


import os, sys
import commune
from commune.utils import *
import gradio

import shlex
import subprocess
class SubprocessModule(commune.Module):
    subprocess_map = {}

    def __init__(self, *args, **kwargs):
        commune.Module.__init__(self, *args,  **kwargs)
        self.load()


    def submit(command):
        return self.run_command(command)
    
    @property
    def load(self):
        self.subprocess_map = self.get_json('subprocess_map')
    def save(self):
        self.put_json('subprocess_map',  self.subprocess_map)

    def rm_subprocess(self, key):
        self.load()
        subprocess_dict = self.subprocess_map[key]
        self.kill_pid(subprocess_dict['pid'])
        del self.subprocess_map[key]
        self.save()
        return pid

    rm = rm_subprocess

    def rm_all(self):
        self.load()
        rm_dict = {}
        for k in self.list_keys():
            rm_dict[k] = self.rm(key=k, load=False, save=False)

        self.save()
        return rm_dict

    def add_subprocess(self, command:str,key=None, cache=True, add_info={}):


        process = subprocess.Popen(shlex.split(command))
        process_state_dict = process.__dict__
        # process_state_dict.pop('_waitpid_lock')

        self.subprocess_map = {k:v for k,v in process_state_dict.items() if k != '_waitpid_lock'}
        if cache == True:
            if key == None or key == 'pid':
                key= str(process.pid)
            self.subprocess_map[key] = add_info

        return self.subprocess_map

    submit = add = add_subprocess  
    
    def ls(self):
        self.load_state()
        return list(self.subprocess_map.keys())

    ls_keys = list_keys = list = ls
   

    @property
    def portConnection( port : int, host='0.0.0.0'):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((host, port))
        if result == 0: return True
        return False

if __name__ == "__main__":
    SubprocessModule()
    # import stre=amlit as st
    # module = SubprocessModule.deploy(actor=False, override={'refresh':True})
    # st.write(module)
    # import ray

    # # st.write(module.subprocess_map)

    # module.client.rest


    # st.write(ray.get(module.ls.remote()))
    # st.write(ray.get(module.rm_all.remote()))
    # st.write(module.add(key='pid', command='python commune/gradio/api/module.py  --module="gradio.client.module.ClientModule"'))
    # st.write(module.getattr('cache'))
    # st.write(ray.get(module.ls.remote()))



