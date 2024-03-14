import os
import subprocess
import argparse

class PoetryEnvManager:
    def __init__(self, env_dir):
        self.env_dir = env_dir

    def destroy_environment(self):
        if os.path.exists(self.env_dir):
            activate_script = os.path.join(self.env_dir, 'Scripts', 'deactivate.bat')
            subprocess.run(f'call {activate_script}', shell=True, check=True)
            print("Virtual environment deactivated successfully.")

            subprocess.run(['poetry', 'env', 'remove', '--all'], cwd=self.env_dir, check=True)
            print("Environment destroyed successfully.")
        else:
            print("Environment doesn't exist.")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Create a new virtual enviroment directory.")
    # parser.add_argument('env_directory', type=str, help='Name of the directory to create')
    # args = parser.parse_args()
    # directory_name = args.env_directory

    env_manager = PoetryEnvManager(".venv")
    
    env_manager.destroy_environment()
