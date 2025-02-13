import commune as c
import subprocess
import requests
import base64
import re


class git(c.Module):

    def __init__(self, repo_url='commune-ai/commune'):
        self.repo_url = repo_url
        self.api_base = "https://api.github.com"
        self.repo_path = self._get_repo_path()
    def is_repo(self, lib_path:str ):
        # has the .git folder
        return c.cmd(f'ls -a {lib_path}').count('.git') > 0

    @staticmethod
    def clone(repo_url:str, target_directory:str = None, branch=None):
        prefix = 'https://github.com/'
        if not repo_url.startswith(prefix):
            repo_url = f'{prefix}{repo_url}'

        if target_directory == None:
            target_directory = repo_url.split('/')[-1].split('.')[0]
        else:
            target_directory = c.resolve_path(target_directory)
        # Clone the repository
        return subprocess.run(['git', 'clone', repo_url, target_directory])

        

    @staticmethod
    def content(url='LambdaLabsML/examples/main/stable-diffusion-finetuning/pokemon_finetune.ipynb', prefix='https://raw.githubusercontent.com'):
        return c.module('web')().page_content(f'{prefix}/{url}')
    
    submodule_path = c.repo_name + '/repos'
    def add_submodule(self, url, name=None, prefix=submodule_path):
        if name == None:
            name = url.split('/')[-1].split('.')[0].lower()
        
        if prefix != None:
            name = f'{prefix}/{name}'

        c.cmd(f'git submodule add {url} {name}')

    @classmethod
    def pull(cls, stash:bool = False, cwd=None):
        if cwd is None:
            cwd = c.lib_path
        if stash:
            c.cmd('git stash', cwd=cwd)
        c.cmd('git pull', cwd=cwd)
        return {'success':True, 'message':'pulled'}

    @classmethod
    def push(cls, msg:str='update', cwd=None):
        if cwd is None:
            cwd = c.lib_path
        c.cmd(f'git add .', cwd=cwd)
        c.cmd(f'git commit -m "{msg}"', bash=True, cwd=cwd)
        c.cmd(f'git push', cwd=cwd)

    @classmethod
    def status(cls, cwd=None):
        if cwd is None:
            cwd = c.lib_path
        return c.cmd(f'git status', cwd=cwd, verbose=False)

        
    def git_repos(self, path='./'):
            import os
            repos = []
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    if d.endswith('.git'):
                        repos +=  [f"{root}"]

            repos = [r for r in repos if not r == path]

            return repos



    @classmethod
    def commit(cls, message='update', push:bool = True):
        c.cmd(f'git commit -m "{message}"')
        if push:
            cls.push()

    @classmethod
    def repo_url(cls, lib_path:str = None) -> str:
        llib_path = cls.resolve_lib_path(lib_path)
        return c.cmd('git remote -v',cwd=lib_path, verbose=False).split('\n')[0].split('\t')[1].split(' ')[0]
    
    @classmethod
    def commit_hash(cls, lib_path:str = None):
        lib_path = cls.resolve_lib_path(lib_path)
        return c.cmd('git rev-parse HEAD', cwd=lib_path, verbose=False).split('\n')[0].strip()

    def reset_hard(self, lib_path:str = None):
        lib_path = self.resolve_lib_path(lib_path)
        return c.cmd('git reset --hard', cwd=lib_path, verbose=False)
    
    def resolve_lib_path(self, lib_path:str = None):
        if lib_path == None:
            lib_path = c.lib_path
        return lib_path
    
    @classmethod
    def merge_remote_repo(cls, remote_name:str, remote_url:str, remote_branch:str, local_branch:str, cwd=None):
        # Add the remote repository
        add_remote_command = f"git remote add {remote_name} {remote_url}"

        # Fetch the contents of the remote repository
        fetch_command = f"git fetch {remote_name}"

        # Checkout to your local branch
        checkout_command = f"git checkout {local_branch}"

        # Merge the remote branch into your local branch
        merge_command = f"git merge {remote_name}/{remote_branch}"

        # Push the changes to your remote repository
        push_command = f"git push origin {local_branch}"

        cmds = [add_remote_command,
                 fetch_command,
                   checkout_command,
                     merge_command, 
                     push_command]
        
        cmd = ' && '.join(cmds)
        return c.cmd(cmd, cwd)
    

    def get_repos(self, username_or_org="openai"):
        import requests
        import re
        # Get the HTML of the repositories page
        url = f"https://github.com/{username_or_org}?tab=repositories"
        # response = requests.get(url)
        return c.module('web')().page_content(url)["links"]
    
        
    def _get_repo_path(self):
        """Extract repository path from URL"""
        return "/".join(self.repo_url.split("github.com/")[1].split("/"))

    def get_file_content(self, path):
        """Get content of a specific file"""
        url = f"{self.api_base}/repos/{self.repo_path}/contents/{path}"
        response = requests.get(url)
        if response.status_code == 200:
            content = response.json()
            if content.get("encoding") == "base64":
                return base64.b64decode(content["content"]).decode()
        return None

    def get_directory_contents(self, path=""):
        """Get contents of a directory"""
        url = f"{self.api_base}/repos/{self.repo_path}/contents/{path}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get contents of {path}") 
            print(response.text)
        return []

    def process_code(self, code_content):
        """Process the code content - example processing"""
        if not code_content:
            return None
        
        # Example processing:
        # 1. Remove comments
        code_content = re.sub(r'#.*', '', code_content)
        code_content = re.sub(r'"""[\s\S]*?"""', '', code_content)
        
        # 2. Remove empty lines
        code_content = "\n".join([line for line in code_content.split("\n") if line.strip()])
        
        return code_content

    def process_repository(self, path=""):
        """Process entire repository recursively"""
        processed_files = {}
        contents = self.get_directory_contents(path)
        
        for item in contents:
            if isinstance(item, dict):
                if item["type"] == "file" and item["name"].endswith(".py"):
                    content = self.get_file_content(item["path"])
                    processed_content = self.process_code(content)
                    processed_files[item["path"]] = processed_content
                elif item["type"] == "dir":
                    sub_processed = self.process_repository(item["path"])
                    processed_files.update(sub_processed)
                    
        return processed_files

    @classmethod
    def test(cls):
        # Initialize processor
        processor = cls("https://github.com/commune-ai/eliza")
        
        # Process repository
        processed_files = processor.process_repository()
        
        # Print results
        file2content = {file_path: processed_content if processed_content else "No content" for file_path, processed_content in processed_files.items()}
        for file_path, processed_content in processed_files.items():
            print(f"\nFile: {file_path}")
            print("Processed content:")
            print(processed_content[:200] + "..." if processed_content else "No content")