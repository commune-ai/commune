import commune as c
import os

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