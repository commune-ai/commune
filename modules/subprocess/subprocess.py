


import os, sys
import shlex
import subprocess
import socket
import commune as c

class SubprocessModule(c.Module):
    subprocess_map = {}
    def kill(self, key):
        subprocess_dict = self.subprocess_map[key]
        pid = subprocess_dict['pid']
        try:
            self.kill_pid(pid)
        except ProcessLookupError:
            pass
        self.pop_cache(key)
        
        return pid

    def kill_all(self):
        rm_dict = {}
        for k in self.list_keys():
            rm_dict[k] = self.rm(key=k, load=False, save=False)

        return rm_dict
    
    def remote_fn(self, key, fn, *args, **kwargs):
        subprocess_dict = self.get(key)
        pid = subprocess_dict['pid']
        return self.remote_fn_pid(pid, fn, *args, **kwargs)

    def serve(self, command:str,cache=True, add_info={}, key=None ):

        process = subprocess.Popen(shlex.split(command))
        process_state_dict = process.__dict__
        # process_state_dict.pop('_waitpid_lock')

        subprocess_dict = {k:v for k,v in process_state_dict.items() if k != '_waitpid_lock'}
        if cache:
            if key == None or key == 'pid':
                key= str(process.pid)
            subprocess_dict = (subprocess_dict, add_info)
            self.put_json(key, subprocess_dict)

        return {'pid':process.pid, 'command':command, 'key':key, 'add_info':add_info}
    
    def ls(self):
        self.load_state()
        return list(self.subprocess_map.keys())

    ls_keys = list_keys = list = ls
   


SubprocessModule.run(__name__)