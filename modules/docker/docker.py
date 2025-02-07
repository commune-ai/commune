
import os
import pandas as pd
from typing import List, Dict, Union
import commune as c

class Docker(c.Module):
    
    def file(self, path = './'): 
        files = self.files(path)
        if len(files) > 0:
            return c.get_text(files[0])
        return {'msg': f'no dockerfile founder in {path}'}

    def files(self, path='./'):
        return [f for f in c.files(path) if f.endswith('Dockerfile')]

    def resolve_repo_path(self, path):
        if path is None:
            path = c.repo_name
        else:
            if not path.startswith('/') or not path.startswith('~') or not path.startswith('.'):
                path = c.repo_name + '/' + path
            else:
                path = os.path.abspath(path)
        return path

    def build(self, path = None , tag = None , sudo=False, verbose=True, no_cache=False, env={}):
        path = c.resolve_path(path)
        
        if tag is None:
            tag = path.split('/')[-2]

        cmd = f'docker build -t {tag} .'
        if no_cache:
            cmd += ' --no-cache'
        return c.cmd(cmd, sudo=sudo, env=env,cwd=os.path.dirname(path),  verbose=verbose)
    
    def kill(self, name, sudo=False, verbose=True, prune=False):
        # 
        c.cmd(f'docker kill {name}', sudo=sudo, verbose=verbose)
        c.cmd(f'docker rm {name}', sudo=sudo, verbose=verbose)
        if prune:
            c.cmd('docker container prune', sudo=sudo, verbose=verbose)
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
            
    
    def run(self, 
                    path : str = './',
                    cmd : str  = None,
                    volumes:List[str] = None,
                    name: str = None,
                    gpus:list=False,
                    shm_size : str='100g',
                    sudo:bool = False,
                    build:bool = True,
                    ports:Dict[str, int]=None,
                    net : str = 'host',
                    daemon:bool = True,
                    cwd = None,
                    run: bool = True):
        
        '''
        Arguments:

        '''
        name2file = self.name2file(path)
        file2name = self.file2name(path)
        if not 'Dockerfile' in path:
            path = self.files(path)[0]
        if path in file2name:
            image = file2name[path]
            cwd = cwd or os.path.dirname(path)
        else:
            cwd = cwd or c.pwd()
            image = path
        
        name = name or image
        dcmd = f'docker run'
    
        if daemon:
            dcmd += ' -d'


        dcmd += f' --net {net}'

        if build:
            self.build(image, tag=name)
        

        if isinstance(gpus, list):
            gpus = ','.join(map(str, gpus))  
            dcmd += f' --gpus \'"device={gpus}"\''   
        elif isinstance(gpus, str):
            dcmd += f' --gpus "{gpus}"'
        else:
            pass

        if shm_size != None:
            dcmd += f' --shm-size {shm_size}'
        
        if ports != None:
            if isinstance(ports, str):
                ports  = [ports]
            elif isinstance(ports, dict):
                ports = [f'{k}:{v}' for k, v in ports.items()]
            
            ports = ' '.join([f'-p {p}' for p in ports])
            dcmd += f' {ports}'

        # ADD THE VOLUMES
        if volumes != None:
            if isinstance(volumes, str):
                volumes = [volumes]
            elif isinstance(volumes, dict):
                volumes = [f'{k}:{v}' for k, v in volumes.items()]
            else: 
                raise Exception(f'{volumes} not supported')
            volumes = ' '.join([f'-v {v}' for v in volumes])
            dcmd += f' {volumes}'
        dcmd += f' --name {name} {image}'
        if cmd is not None:
            dcmd += f' bash -c "{cmd}"'
        return {'cmd': dcmd, 'cwd': cwd}

    
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

    
    def name2file(self, path = None):
       return {l.split('/')[-2] if len(l.split('/'))>1 else c.lib:l for l in self.files(path)}
    
    
    def file2name(self, path = None):
       return {v:k for k,v in self.name2file(path).items()}
    
    
    def resolve_file(self, name):
        if name == None:
            name = 'commune'
        
        if c.path_exists(name):
            return name
        name2file = self.name2file()
        if name in name2file:
            return name2file[name]
        else:
            raise ValueError(f'Could not find docker file for {name}')
        
    get_file = resolve_file


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

    def files(self, path = None):
        if path is None:
            path = c.libpath + '/'
        files = []
        for l in c.walk(path):
            if l.endswith('Dockerfile'):
                files.append(l)
        return files
    
    def name2file(self, path = None):
        if path is None:
            path = self.libpath + '/'
        return {l.split('/')[-2] if len(l.split('/'))>1 else c.lib:l for l in self.files(path)}

    def prune(self):
        return c.cmd('docker container prune')

    def start_docker(self):
        return c.cmd('systemctl start docker')
