import commune as c
import os
import pandas as pd
import json

class Docker(c.Module): 

    def ps(self, sudo=False):
        return c.cmd('docker ps -a', sudo=sudo)
    @classmethod
    def dockerfile(cls, path = c.repo_path): 
        path =  [f for f in c.ls(path) if f.endswith('Dockerfile')][0]
        return c.get_text(path)
    
    @classmethod
    def resolve_repo_path(cls, path):
        if path is None:
            path = c.repo_path
        else:
            path = c.repo_path + '/' + path
        return path

    @classmethod
    def resolve_docker_compose_path(cls,path = None):
        path = cls.resolve_repo_path(path)
        return [f for f in c.ls(path) if 'docker-compose' in os.path.basename(f)][0]
        return path

    @classmethod
    def docker_compose(cls, path = c.repo_path): 
        docker_compose_path = cls.resolve_docker_compose_path(path)
        return c.load_yanl(docker_compose_path)
    
    @classmethod
    def build(cls, path = None, tag = None, sudo=False):
        path = cls.resolve_repo_path(path)
        return c.cmd(f'docker-compose build', sudo=sudo, cwd=path)
    

    @classmethod
    def rm_sudo(cls, sudo:bool=True, verbose:bool=True):
        '''
        To remove the requirement for sudo when using Docker, you can configure Docker to run without superuser privileges. Here's how you can do it:
        Create a Docker group (if it doesn't exist) and add your user to that group:
        bash
        Copy code
        sudo groupadd docker
        sudo usermod -aG docker $USER
        return c.cmd(f'docker rm -f {name}', sudo=True)
        '''
        c.cmd(f'groupadd docker', sudo=sudo, verbose=verbose)
        c.cmd(f'usermod -aG docker $USER', sudo=sudo, verbose=verbose)
        c.cmd(f'chmod 666 /var/run/docker.sock', sudo=sudo, verbose=verbose)



    
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



    