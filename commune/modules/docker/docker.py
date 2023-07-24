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
    
    def launch(self, 
                    image : str,
                    tag: str = None,
                    gpus:list='all',
                    shm_size : str='100g',
                    volume:str = 'data',
                    build:bool = True,
                    max_shard_ratio = 0.5,
                    refresh:bool = False,
                    sudo:bool = False,
                    port:int=None,
                    volumes = None,
                    internal_port:int=None):

        if tag != None:
            tag = str(tag)
        if build:
            self.build(image)

        cmd = f'docker run'

        if deamon:
            cmd += '-d'

        # ADD THE GPUS
        if gpus == None:
            gpus = c.gpus()
        gpus = ','.join(map(str, gpus))     
        if gpus != None:
            cmd += f' --gpus device={gpus}'
        
        # ADD THE SHM SIZE
        if shm_size != None:
            cmd += f' --shm-size {shm_size}'
        
        # ADD THE PORTS
        if port == None:
            port = c.resolve_port(port)
        if internal_port == None:
            internal_port = port
        if port != None:
            cmd += f' -p {port}:{internal_port}'
            


        # ADD THE VOLUMES
        if volumes is not None:
            if isinstance(volumes, list):
                volumes = {v:v for v in volumes}
            for v_from, v_to in volumes.items():
                cmd += f'-v {v_from}:{v_to}'

        cmd += '--name {name} {self.image} {cmd_args}'

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
       



    
    def psdf(self,load=True, save=False, keys = [ 'container_id', 'names', 'ports'], idx_key ='container_id'):
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
                rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        df['ports'] = df['ports'].apply(lambda x: x.split('->')[0].strip() if len(x.split('->')) > 1 else x)
        df = df[keys]
        df.set_index(idx_key, inplace=True)
        return df   
    def ps(self):
        df = self.psdf()
        return self.psdf()['names'].tolist()
    


    @classmethod
    def dockerfiles(cls, path = None):
       if path is None:
           path = c.libpath + '/'
       return [l.replace(path, '') for l in c.walk(path) if l.endswith('Dockerfile')]
    

    @classmethod
    def dockercomposefiles(cls, path = None):
       if path is None:
           path = c.libpath + '/'
       return [l.replace(path, '') for l in c.walk(path) if l.endswith('docker-compose.yaml') or l.endswith('docker-compose.yml')]