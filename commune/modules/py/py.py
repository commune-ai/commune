import os
import sys
import subprocess
import shutil
import commune as c

class PythonEnvManager(c.Module):
    def __init__(self, base_path=None):
        self.base_path = base_path or os.path.expanduser('~')
        self.venv_path = os.path.join(self.base_path, '.envs')
        os.makedirs(self.venv_path, exist_ok=True)

    def create_env(self, env):
        env_path = os.path.join(self.venv_path, env)
        if os.path.exists(env_path):
            print(f"Environment {env} already exists.")
            return
        subprocess.check_call([sys.executable, '-m', 'venv', env_path])
        return {'msg': f"Created environment {env} at {env_path}"}
    create = create_env

    def remove_env(self, env):
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
    
    def envs_paths(self):
        return list(self.env2path().values())

    def packages(self, env, search=None):
        '''Available environments:'''
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts', 'python')
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        output = subprocess.check_output([env_path, '-m', 'pip', 'list']).decode('utf-8')
        output =  {line.split(' ')[0]:line.split(' ')[-1] for line in output.split('\n')[2:-1]}

        if search:
            output = {k:v for k,v in output.items()}
        
        return output
    
    def run(self, script_path, env=None):
        if not env:
            env = list(self.envs())[0]
        env_path = os.path.join(self.venv_path, env, 'bin' if os.name == 'posix' else 'Scripts')
        if not os.path.exists(env_path):
            print(f"Environment {env} does not exist.")
            return
        activation_script = os.path.join(env_path, 'activate') if os.name == 'posix' else os.path.join(env_path, 'Scripts', 'activate.bat')
        python_executable = os.path.join(env_path, 'python') if os.name == 'posix' else os.path.join(env_path, 'python.exe')

        os.system(f"{'source ' if os.name == 'posix' else ''}{activation_script}")
        os.system(python_executable + ' ' + script_path)
        if os.name == 'posix':
            os.system("deactivate")
        else:
            os.system(f"{'cd..' if os.path.dirname(env_path) != os.path.dirname(os.getcwd()) else ''}")

    def env2cmd(self):
        env2cmd = {}
        for env, path in self.env2path().items():
            env2cmd[env] = f'source {path}/bin/activate'
        return env2cmd
    
    def enter_env(self, env):
        cmd = self.env2cmd().get(env)
        # Add shell interpreter
        cmd = f'bash -c "{cmd}"'
        # Print the command for debugging purposes
        print(cmd)
        # Execute the command
        return os.system(cmd)