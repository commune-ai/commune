


import os, sys
from commune import Module
from commune.utils import *
import shlex
import subprocess
import socket

class SubprocessModule(Module):
    subprocess_map = {}
    def __init__(self, config=None, **kwargs):
        Module.__init__(self, config=config)
        self.subprocess_map_path = self.cache_path

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.config,)
        return deserializer, serialized_data
    
    @property
    def subprocess_map(self):
        self.load_cache()
        return self.cache

    def rm_subprocess(self, key):
        subprocess_dict = self.subprocess_map[key]
        pid = subprocess_dict['pid']
        try:
            self.kill_pid(pid)
        except ProcessLookupError:
            pass
        self.pop_cache(key)
        
        return pid

    rm = rm_subprocess

    def rm_all(self):
        rm_dict = {}
        for k in self.list_keys():
            rm_dict[k] = self.rm(key=k, load=False, save=False)

        return rm_dict

    def add_subprocess(self, command:str,key=None, cache=True, add_info={}):

        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        process_state_dict = process.__dict__
        # process_state_dict.pop('_waitpid_lock')
        subprocess_dict = {k:v for k,v in process_state_dict.items() if k != '_waitpid_lock'}
        if cache == True:
            if key == None or key == 'pid':
                key= str(process.pid)
            self.put_cache(key, subprocess_dict)
        return subprocess_dict

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

    @classmethod
    def streamlit(cls):
        st.write(cls().subprocess_map)

if __name__ == "__main__":
    SubprocessModule.run()
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



