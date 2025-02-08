import os
import sys
import subprocess
import shutil
import commune as c

class Py(c.Module):
    def __init__(self, venv_path=None):
        self.set_venv_path(venv_path)

    def set_venv_path(self, venv_path):
        self.venv_path = venv_path or os.path.expanduser('~/.envs')
        os.makedirs(self.venv_path, exist_ok=True)

    def create(self, env):
        env_path = os.path.join(self.venv_path, env)
        if os.path.exists(env_path):
            print(f"Environment {env} already exists.")
            return
        subprocess.check_call([sys.executable, '-m', 'venv', env_path])
        return {'msg': f"Created environment {env} at {env_path}"}

    def remove(self, env):
        env_path = os.path.join(self.venv_path, env)
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        shutil.rmtree(env_path)
        return dict(msg=f"Deleted environment {env}")

    def install(self, env, package_name):
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts', 'python')
        initial_args = [env_path, '-m', 'pip', 'install']
        if os.path.exists(package_name):
            initial_args += ['-e']
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        subprocess.check_call([*initial_args, package_name])
        return dict(msg=f"Installed {package_name} in environment {env}")

    def env2path(self):
        env_paths =  c.ls(self.venv_path)
        return {v.split('/')[-1] : v for v in env_paths}
    

    def envs(self):
        return list(self.env2path().keys())
    def env2libs(self):
        return {env:self.libs(env) for env in self.envs()}
    
    
    def envs_paths(self):
        return list(self.env2path().values())

    def libs(self, env=None, search=None):
        '''Available environments:'''
        env = self.get_env(env)
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts', 'python')
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        output = subprocess.check_output([env_path, '-m', 'pip', 'list']).decode('utf-8')
        output =  {line.split(' ')[0]:line.split(' ')[-1] for line in output.split('\n')[2:-1]}

        if search:
            output = {k:v for k,v in output.items()}
        
        return output
    
    def get_env(self, env):
        envs = self.envs()
        if not env:
            env = envs[0]
        assert env in envs, f"Environment {env} does not exist, create one using `create_env`"
        print('Selecting environment')
        return env
    
    def get_activation_path(self, env):
        env = self.get_env(env)
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts')
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        activation_script = os.path.join(env_path, 'activate') if os.name == 'posix' else os.path.join(env_path, 'Scripts', 'activate.bat')
        return activation_script
    

    def run(self, path=c.repo_name+'/modules/sandbox.py', env="bt"):
        env = self.get_env(env)
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts')
        # run the script with the selected environment without activating it
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        cmd = f'{env_path}/python {path}'
        os.system(cmd)
    
    def env2cmd(self):
        env2cmd = {}
        for env, path in self.env2path().items():
            env2cmd[env] = f'{path}/bin/activate'
        return env2cmd
            
    def enter(self, env):
        env = self.get_env(env)
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts')
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        activation_script = os.path.join(env_path, 'activate') if os.name == 'posix' else os.path.join(env_path, 'Scripts', 'activate.bat')
        os.system(f"{'source ' if os.name == 'posix' else ''}{activation_script}")
        return dict(msg=f"Entered environment {env}")