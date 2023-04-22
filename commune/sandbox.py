import argparse
import os
import subprocess

def remove_submodule(submodule_name, submodule_path):
    # Remove submodule section from .gitmodules
    with open('.gitmodules', 'r') as file:
        content = file.readlines()

    with open('.gitmodules', 'w') as file:
        for line in content:
            if submodule_name not in line and submodule_path not in line:
                file.write(line)

    # Remove submodule section from .git/config
    git_config_path = os.path.join('.git', 'config')
    with open(git_config_path, 'r') as file:
        content = file.readlines()

    with open(git_config_path, 'w') as file:
        for line in content:
            if submodule_name not in line and submodule_path not in line:
                file.write(line)

    # Remove submodule files and commit changes
    subprocess.run(['git', 'rm', '--cached', submodule_path])
    subprocess.run(['git', 'add', '.gitmodules'])
    subprocess.run(['git', 'commit', '-m', f'Removed {submodule_name} submodule'])

    # Delete submodule directory
    subprocess.run(['rm', '-rf', submodule_path])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove a submodule from a GitHub repository')
    parser.add_argument('name', help='Name of the submodule to remove')
    parser.add_argument('path', help='Path of the submodule to remove')
    args = parser.parse_args()

    remove_submodule('subtensor', 'subtensor')
