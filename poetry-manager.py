import os
import subprocess

class PoetryEnvManager:
    def __init__(self, env_dir):
        self.env_dir = env_dir

    def create_virtual_environment(self):
        subprocess.run(['python', '-m', 'venv', '.venv' ], check=True)
        print("Virtual Environment created successfully.")


    def create_environment(self):
        subprocess.run(['poetry', 'install'], check=True)
        print("Environment created successfully.")

    def destroy_environment(self):
        if os.path.exists(self.env_dir):
            subprocess.run(['poetry', 'env', 'remove', '.venv'], check=True)
            print("Environment destroyed successfully.")
        else:
            print("Environment doesn't exist.")

    def list_environments(self):
        envs = subprocess.run(['poetry', 'env', 'list'], capture_output=True, text=True).stdout
        print(envs)


if __name__ == "__main__":

    env_manager = PoetryEnvManager(".venv")
    
    env_manager.create_virtual_environment()
    env_manager.list_environments()
    # env_manager.create_environment()

    # env_manager.destroy_environment()
