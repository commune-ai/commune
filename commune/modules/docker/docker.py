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
    def ps(cls,  sudo:bool = False):
        data = [f for f in c.cmd('docker ps', sudo=sudo, verbose=False).split('\n')[1:]]
        def parse_container_info(container_str):
            container_info = {}
            fields = container_str.split()

            container_info['container_id'] = fields[0]
            container_info['image'] = fields[1]
            container_info['command'] = fields[2]
            container_info['created'] = fields[3] + ' ' + fields[4]
            container_info['status'] = ' '.join(fields[5:fields.index('ago') + 1])
            container_info['ports'] = ' '.join(fields[fields.index('ago') + 2:-1])
            container_info['name'] = fields[-1]

            return container_info

        
        return [parse_container_info(container_str) for container_str in data if container_str]


    @classmethod
    def containers(cls,  sudo:bool = False):
        return [container['name'] for container in cls.ps(sudo=sudo)]
    
    @classmethod 
    def chmod_scripts(cls):
        c.cmd(f'bash -c "chmod +x {c.libpath}/scripts/*"', verbose=True)

    def install_gpus(self):
        self.chmod_scripts
        c.cmd('./scripts/nvidia_docker_setup.sh', cwd=self.libpath, verbose=True)


    def build_commune(self, sudo=False):
        self.build(path=self.libpath, sudo=sudo)

    @classmethod
    def build(cls, tag = None,  path = None  , sudo=False):
        if path is None:
            path = c.libpath
            tag = c.libpath.split('/')[-1]
        assert tag is not None, 'tag must be specified'

        path = c.resolve_path(path)
        cmd = f'docker build -t {tag} .'

        c.print(path, cmd)
        c.cmd(cmd, cwd=path, verbose=True, sudo=sudo, bash=True)
    
    def launch(self, model :str = None,
                    tag: str = None,
                    num_shard:int=None, 
                    gpus:list='all',
                    shm_size : str='100g',
                    volume:str = 'data',
                    build:bool = True,
                    max_shard_ratio = 0.5,
                    refresh:bool = False,
                    sudo = False,
                    port=None):

        if model == None:
            model = self.config.model
        if tag != None:
            tag = str(tag)
        name =  (self.image +"_"+ model) + ('_'+tag if tag  else '')
        if self.server_exists(name) and refresh == False:
            c.print(f'{name} already exists')
            return

        if build:
            self.build()

        if gpus == None:
            gpus = c.model_max_gpus(model)
        
        num_shard = len(gpus)
        gpus = ','.join(map(str, gpus))

        c.print(f'gpus: {gpus}')
        
        model_id = self.config.shortcuts.get(model, model)
        if port == None:
            port = c.resolve_port(port)

        volume = self.resolve_path(volume)
        if not c.exists(volume):
            c.mkdir(volume)

        cmd_args = f'--num-shard {num_shard} --model-id {model_id}'



        cmd = f'docker run -d --gpus device={gpus} --shm-size {shm_size} -p {port}:80 -v {volume}:/data --name {name} {self.image} {cmd_args}'

        c.print(cmd)
        output_text = c.cmd(cmd, sudo=sudo, output_text=True)

        if 'Conflict. The container name' in output_text:
            c.print(f'container {name} already exists, restarting...')
            contianer_id = output_text.split('by container "')[-1].split('". You')[0].strip()
            c.cmd(f'docker rm -f {contianer_id}', sudo=sudo, verbose=True)
            c.cmd(cmd, sudo=sudo, verbose=True)
        else: 
            c.print(output_text)


        self.update()
       



    