import commune as c


class Git(c.Module):
    
    
    @staticmethod
    def clone(repo_url:str, target_directory:str = None):
        if target_directory == None:
            target_directory = repo_url.split('/')[-1].split('.')[0]
        else:
            target_directory = c.resolve_path(target_directory)
        import subprocess

        # Clone the repository
        subprocess.run(['git', 'clone', repo_url, target_directory])

        # Remove the .git directory
        subprocess.run(['rm', '-rf', f'{target_directory}/.git'])
        

    @staticmethod
    def content(url='LambdaLabsML/examples/main/stable-diffusion-finetuning/pokemon_finetune.ipynb',
                   prefix='https://raw.githubusercontent.com'):
        return c.module('tool.web').rget(url=f'{prefix}/{url}')
    
    submodule_path = c.repo_path + '/repos'
    def add_submodule(self, url, name=None, prefix=submodule_path):
        if name == None:
            name = url.split('/')[-1].split('.')[0].lower()
        
        if prefix != None:
            name = f'{prefix}/{name}'

        c.cmd(f'git submodule add {url} {name}')

    addsub = add_submodule

    @classmethod
    def pull(cls, stash:bool = False, cwd=None):
        if cwd is None:
            cwd = c.libpath
        if stash:
            c.cmd('git stash', cwd=cwd)
        c.cmd('git pull', cwd=cwd)
        return {'success':True, 'message':'pulled'}

    @classmethod
    def push(cls, msg:str='update', cwd=None):
        if cwd is None:
            cwd = c.libpath
        c.cmd(f'git add .', cwd=cwd)
        c.cmd(f'git commit -m "{msg}"', bash=True, cwd=cwd)
        c.cmd(f'git push', cwd=cwd)

    @classmethod
    def gstat(cls, cwd=None):
        if cwd is None:
            cwd = c.libpath
        return c.cmd(f'git status', cwd=cwd, verbose=False)

        


    @classmethod
    def commit(cls, message='update', push:bool = True):
        c.cmd(f'git commit -m "{message}"')
        if push:
            cls.push()

    @classmethod
    def repo_url(cls, libpath:str = None) -> str:
        libpath = libpath if libpath else c.libpath
        return c.cmd('git remote -v',cwd=libpath, verbose=False).split('\n')[0].split('\t')[1].split(' ')[0]
    
    @classmethod
    def commit_hash(cls, libpath:str = None):
        libpath = libpath if libpath else c.libpath
        return c.cmd('git rev-parse HEAD', cwd=libpath, verbose=False).split('\n')[0].strip()

