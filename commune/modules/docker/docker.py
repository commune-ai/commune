import commune as c
import os
import pandas as pd
import json

class Docker(c.Module): 

    def ps(sudo=False):
        return c.cmd('docker ps -a', sudo=sudo)

    def ps():
        return c.cmd('sudo docker ps -a')
    
    @classmethod
    def dockerfile(cls, path = c.repo_path): 
        return [f for f in c.ls(path) if f.endswith('Dockerfile')][0]

    @classmethod
    def docker_compose(cls, path = c.repo_path): 
        return [f for f in c.ls(path) if 'docker-compose' in os.path.basename(f)][0]
    

    def build(path = c.repo_path, tag = None, sudo=True):
        
        if tag is None:
            tag = os.path.basename(path)
        return c.cmd(f'docker-compose build', sudo=sudo, cwd=path)
    
    
    @classmethod
    def containers(cls,  sudo:bool = True):
        data = [f for f in c.cmd('docker ps -a', sudo=sudo).split('\n')[1:]]
        def parse_container_info(container_str):
            container_info = {}
            fields = container_str.split()

            c.print(fields)
            container_info['container_id'] = fields[0]
            container_info['image'] = fields[1]
            container_info['command'] = fields[2]
            container_info['created'] = fields[3] + ' ' + fields[4]
            container_info['status'] = ' '.join(fields[5:fields.index('ago') + 1])
            container_info['ports'] = ' '.join(fields[fields.index('ago') + 2:-1])
            container_info['name'] = fields[-1]

            return container_info

        
        return [parse_container_info(container_str) for container_str in data if container_str]
