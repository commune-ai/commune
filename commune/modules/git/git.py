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
    def gitcontent(url='LambdaLabsML/examples/main/stable-diffusion-finetuning/pokemon_finetune.ipynb',
                   prefix='https://raw.githubusercontent.com'):
        return c.module('web').rget(url=f'{prefix}/{url}')
    
    
        
    