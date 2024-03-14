import os
import subprocess
import argparse
import shutil


class PoetryEnvManager:
    def __init__(self, env_dir):
        self.env_dir = env_dir


    def create_virtual_environment(self):

        if os.path.exists(self.env_dir):
            shutil.rmtree(self.env_dir)
        else:
            print('No exist virtual environment.')

        try:
            subprocess.run(['python', '-m', 'venv', self.env_dir ], check=True)
            print("Virtual Environment created successfully.")

            activate_script = os.path.join(self.env_dir, 'Scripts', 'activate.bat')

            subprocess.run(f'call {activate_script}', shell=True, check=True)
            print("Virtual environment activated successfully.")
            
        except Exception as e:
            print(e)

    def create_environment(self):
        subprocess.run(['poetry', 'install'], check=True)
        print("Poetry packages installed successfully.")

    def list_environments(self):
        envs = subprocess.run(['poetry', 'env', 'list'], capture_output=True, text=True).stdout
        print(envs)

    def destroy_environment(self):
        if os.path.exists(self.env_dir):
            shutil.rmtree(self.env_dir)
            print("Environment destroyed successfully.")
        else:
            print('No exist virtual environment.')


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Create a new virtual environent directory.")
    # parser.add_argument('env_directory', type=str, help='Name of the directory to create')
    # args = parser.parse_args()
    # directory_name = args.env_directory

    env_manager = PoetryEnvManager('.venv')
    
    env_manager.create_virtual_environment()
    env_manager.list_environments()
    env_manager.create_environment()
    env_manager.destroy_environment()
