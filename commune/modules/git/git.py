import commune as c


class Git(c.Module):
    
    
    @staticmethod
    def clone(repo_url:str, target_directory:str = None):
        if target_directory == None:
            target_directory = repo_url.split('/')[-1].split('.')[0]
        import subprocess

        # Clone the repository
        subprocess.run(['git', 'clone', repo_url, target_directory])

        # Remove the .git directory
        subprocess.run(['rm', '-rf', f'{target_directory}/.git'])
        
    