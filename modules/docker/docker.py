
import os
import pandas as pd
from typing import List, Dict, Union
import commune as c

class Docker:
    
    def dockerfile(self, path = c.reponame): 
        path =  [f for f in c.ls(path) if f.endswith('Dockerfile')][0]
        return c.get_text(path)
    
    def resolve_repo_path(self, path):
        if path is None:
            path = c.reponame
        else:
            if not path.startswith('/') or not path.startswith('~') or not path.startswith('.'):
                path = c.reponame + '/' + path
            else:
                path = os.path.abspath(path)
        return path

    def resolve_docker_compose_path(self,path = None):
        path = self.resolve_repo_path(path)
        return [f for f in c.ls(path) if 'docker-compose' in os.path.basename(f)][0]

    
    def docker_compose(self, path = c.reponame): 
        docker_compose_path = self.resolve_docker_compose_path(path)
        return c.load_yanl(docker_compose_path)

    
    def resolve_docker_path(self, path = None):
        path = self.resolve_repo_path(path)
        return [f for f in c.ls(path) if 'Dockerfile' in os.path.basename(f)][0]
    
    
    def build(self, path = None , tag = None , sudo=False, verbose=True, no_cache=False, env={}):
        path = c.resolve_path(path)
        
        if tag is None:
            tag = path.split('/')[-2]

        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
        return c.cmd(cmd, sudo=sudo, env=env,cwd=os.path.dirname(path),  verbose=verbose)
    
    def kill(self, name, sudo=False, verbose=True, prune=False):
        c.cmd(f'docker kill {name}', sudo=sudo, verbose=verbose)
        c.cmd(f'docker rm {name}', sudo=sudo, verbose=verbose)
        if prune:
            c.cmd('docker container prune', sudo=sudo, verbose=verbose)
        return {'status': 'killed', 'name': name}

    
    def kill_many(self, name, sudo=False, verbose=True):
        servers = self.ps(name)
        for server in servers:
            self.kill(server, sudo=sudo, verbose=verbose)
            c.print(f'killed {server}', verbose=verbose)
        return {'status': 'killed', 'name': name}

    
    def kill_all(self, sudo=False, verbose=True):
        servers = self.ps()
        for server in servers:
            self.kill(server, sudo=sudo, verbose=verbose)
            c.print(f'killed {server}', verbose=verbose)
        return {'status': 'killed'}
    
    def rm(self, name, sudo=False, verbose=True):
        c.cmd(f'docker rm {name}', sudo=sudo, verbose=verbose)
        return {'status': 'removed', 'name': name}

    
    def exists(self, name:str):
        return name in self.ps()

    
    def rm_sudo(self, sudo:bool=True, verbose:bool=True):
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



    

    
    def containers(self,  sudo:bool = False):
        return [container['name'] for container in self.ps(sudo=sudo)]
    
     
    def chmod_scripts(self):
        c.cmd(f'bash -c "chmod +x {c.libpath}/scripts/*"', verbose=True)



    def install_gpus(self):
        self.chmod_scripts()
        c.cmd('./scripts/nvidia_docker_setup.sh', cwd=c.libpath, verbose=True,bash=True)

    def install(self):
        self.chmod_scripts()
        c.cmd('./scripts/install_docker.sh', cwd=c.libpath, verbose=True,bash=True)

    def install_docker_compose(self, sudo=False):
        return c.cmd('apt install docker-compose', verbose=True, sudo=True)
    # def build_commune(self, sudo=False):
    #     self.build(path=self.libpath, sudo=sudo)

    
    def images(self, to_records=True):
        text = c.cmd('docker images', verbose=False)
        df = []
        cols = []
        for i, l in enumerate(text.split('\n')):
            if len(l) > 0:
                if i == 0:
                    cols = [_.strip().replace(' ', '_').lower() for _ in l.split('  ') if len(_) > 0]
                else:
                    df.append([_.strip() for _ in l.split('  ') if len(_) > 0])
        df = pd.DataFrame(df, columns=cols) 
        if to_records:
            return df.to_records()
        return df
    
    def rm_image(self, image_id):
        response = {'success': False, 'image_id': image_id}
        c.cmd(f'docker image rm -f {image_id}', verbose=True)
        response['success'] = True
        return response

    def rm_images(self, search:List[str]=None):
        image_records = self.images(to_records=False)
        responses = []
        for i, image_record in image_records.iterrows():
            image_dict = image_record.to_dict()

            if search == None or str(search.lower()) in image_dict['repository']:
                r = self.rm_image(image_dict['image_id'])
                responses.append(r)
                
        return {'success': True, 'responses': responses }
    

    
    def image2id(self, image=None):
        image2id = {}
        df = self.images()
        for  i in range(len(df)):
            image2id[df['REPOSITORY'][i]] = df['IMAGE_ID'][i]
        if image != None:
            id = image2id[image]
        return id
            
    
    def deploy(self, 
                    image : str,
                    cmd : str  = 'ls',
                    volumes:List[str] = None,
                    name: str = None,
                    gpus:list=False,
                    shm_size : str='100g',
                    sudo:bool = False,
                    build:bool = True,
                    ports:Dict[str, int]=None,
                    net : str = 'host',
                    daemon:bool = True,
                    run: bool = True):
        
        '''
        Arguments:

        '''
        if name is None:
            name = image

        docker_cmd = f'docker run'

        docker_cmd += f' --net {net} '

        if build:
            self.build(image, tag=name)
        
        if daemon:
            docker_cmd += ' -d '

        if isinstance(gpus, list):
            gpus = ','.join(map(str, gpus))  
            docker_cmd += f' --gpus \'"device={gpus}"\''   
        elif isinstance(gpus, str):
            docker_cmd += f' --gpus "{gpus}"'
        else:
            pass
        
        # ADD THE SHM SIZE
        # what is this?
        if shm_size != None:
            docker_cmd += f' --shm-size {shm_size}'
        
        if ports != None:
            for external_port, internal_port in ports.items():
                docker_cmd += f' -p {external_port}:{internal_port}'

        # ADD THE VOLUMES
        if volumes is not None:
            if isinstance(volumes, str):
                volumes = [volumes]
            if isinstance(volumes, list):
                docker_cmd += ' '.join([f' -v {v}' for v in volumes])
            elif isinstance(volumes, dict):
                for v_from, v_to in volumes.items():
                    docker_cmd += f' -v {v_from}:{v_to}'

        docker_cmd += f' --name {name} {image}'

        if cmd is not None:
            docker_cmd += f' bash -c "{cmd}"'
        
        c.print(docker_cmd)
        # text_output =  c.cmd(docker_cmd, sudo=sudo, output_text=True)

        # if 'Conflict. The container name' in text_output:
        #     contianer_id = text_output.split('by container "')[-1].split('". You')[0].strip()
        #     c.cmd(f'docker rm -f {contianer_id}', verbose=True)
        #     text_output = c.cmd(docker_cmd, verbose=True)
        # self.update()
       
    
    def psdf(self, load=True, save=False, idx_key ='container_id'):
        output_text = c.cmd('docker ps', verbose=False)

        rows = []
        for i, row in enumerate(output_text.split('\n')[:-1]):
            if i == 0:
                columns = [l.lower().strip().replace(' ', '_') for l in row.split('   ') if len(l) > 0]
            else:
                NA_SPACE = "           "
                if len(row.split(NA_SPACE)) > 1:
                    row_splits = row.split(NA_SPACE)
                    row = row_splits[0] + '  NA  ' + ' '.join(row_splits[1:])
                row = [_.strip() for _ in row.split('  ') if len(_) > 0]
                if len(row) == len(columns):
                    rows.append(row)
                else:
                    c.print(rows)

        df = pd.DataFrame(rows, columns=columns)
        df.set_index(idx_key, inplace=True)
        return df   

    
    def ps(self, search = None, df:bool = False):

        psdf = self.psdf()
        paths =  psdf['names'].tolist()
        if search != None:
            paths = [p for p in paths if p != None and search in p]
        if df:
            return psdf
        paths = sorted(paths)
        return paths

    
    def name2dockerfile(self, path = None):
       return {l.split('/')[-2] if len(l.split('/'))>1 else c.lib:l for l in self.dockerfiles(path)}
    
    
    def resolve_dockerfile(self, name):
        if name == None:
            name = 'commune'
        
        if c.exists(name):
            return name
        name2dockerfile = self.name2dockerfile()
        if name in name2dockerfile:
            return name2dockerfile[name]
        else:
            raise ValueError(f'Could not find docker file for {name}')
        
    get_dockerfile = resolve_dockerfile

    
    def compose_paths(self, path = None):
       if path is None:
           path = c.libpath + '/'
       return [l for l in c.walk(path) if l.endswith('docker-compose.yaml') or l.endswith('docker-compose.yml')]
    
    
    def name2compose(self, path=None):
        compose_paths = self.compose_paths(path)
        return {l.split('/')[-2] if len(l.split('/'))>1 else c.lib:l for l in compose_paths}
    
    
    def get_compose_path(self, path:str):
        path = self.name2compose().get(path, path)
        return path

    
    def get_compose(self, path:str):
        path = self.get_compose_path(path)
        return c.load_yaml(path)

    
    def put_compose(self, path:str, compose:dict):
        path = self.get_compose_path(path)
        return c.save_yaml(path, compose)
    

    # 
    # def down(self, path='frontend'):
    #     path = self.get_compose_path(path)
    #     return c.cmd('docker-compose -f {path} down', verbose=True)

    
    def compose(self, 
                path: str,
                compose: Union[str, dict, None] = None,
                daemon:bool = True,
                verbose:bool = True,
                dash:bool = True,
                cmd : str = None,
                build: bool = False,
                project_name: str = None,
                cwd : str = None,
                down: bool = False
                ):
 

        cmd = f'docker-compose' if dash else f'docker compose'

        path = self.get_compose_path(path)
        if compose == None:
            compose = self.get_compose(path)
        
        if isinstance(path, str):
            compose = self.get_compose(path)
        

        if project_name != None:
            cmd += f' --project-name {project_name}'
        c.print(f'path: {path}', verbose=verbose)
        tmp_path = path + '.tmp'
        cmd +=  f' -f {tmp_path} up'

        if daemon:
            cmd += ' -d'


        c.print(f'cmd: {cmd}', verbose=verbose)
        # save the config to the compose path
        c.print(compose)
        c.save_yaml(tmp_path, compose)
        if cwd is None:
            assert os.path.exists(path), f'path {path} does not exist'
            cwd = os.path.dirname(path)
        if build:
            c.cmd(f'docker-compose -f {tmp_path} build', verbose=True, cwd=cwd)
            
        text_output = c.cmd(cmd, verbose=True)

        if 'Conflict. The container name' in text_output:
            contianer_id = text_output.split('by container "')[-1].split('". You')[0].strip()
            c.cmd(f'docker rm -f {contianer_id}', verbose=True)
            text_output = c.cmd(cmd, verbose=True)

        if "unknown shorthand flag: 'f' in -f" in text_output:
            cmd = cmd.replace('docker compose', 'docker-compose')
            text_output = c.cmd(cmd, verbose=True)

        c.rm(tmp_path)
    
    def rm_container(self, name):
        c.cmd(f'docker rm -f {name}', verbose=True)

    
    def logs(self, name, sudo=False, follow=False, verbose=False, tail:int=2):
        cmd = f'docker  logs {name} {"-f" if follow else ""} --tail {tail}'
        return c.cmd(cmd, verbose=verbose)

    def log_map(self, search=None):
        nodes = self.ps(search=search)
        return {name: self.logs(name) for name in nodes}

    
    def tag(self, image:str, tag:str):
        c.cmd(f'docker tag {image} {tag}', verbose=True)
        c.cmd(f'docker push {tag}', verbose=True)
    
    def login(self, username:str, password:str):
        c.cmd(f'docker login -u {username} -p {password}', verbose=True)

    
    def logout(self, image:str):
        c.cmd(f'docker logout {image}', verbose=True)

    
    def dockerfiles(self, path = None):
        if path is None:
            path = c.libpath + '/'
        dockerfiles = []
        for l in c.walk(path):
            if l.endswith('Dockerfile'):
                c.print(l)
                dockerfiles.append(l)
        return dockerfiles
    

    def name2dockerfile(self, path = None):
        if path is None:
            path = self.libpath + '/'
        return {l.split('/')[-2] if len(l.split('/'))>1 else c.lib:l for l in self.dockerfiles(path)}
    

    
    def dashboard(self):
        self = self()
        import streamlit as st
        containers = self.psdf()
        name2dockerfile = self.name2dockerfile()
        names = list(name2dockerfile.keys())
        name = st.selectbox('Dockerfile', names)
        dockerfile = name2dockerfile[name]
        dockerfile_text = c.get_text(dockerfile)
        st.code(dockerfile_text)

    def prune(self):
        return c.cmd('docker container prune')

    def start_docker(self):
        return c.cmd('systemctl start docker')
