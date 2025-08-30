
import commune as c
import os
class Vibe: 
    task = 'make a dank tweet about this for the vibes include the path and score it out of 100 vibes and include'

    def forward(self, module:str='module', task=task, update=False):
        assert c.module_exists(module), f'module {module} does not exist'
        code = c.code(module)
        code_hash = c.hash(code)
        path = self.get_path(f'{module}/{code_hash}')
        vibe = c.get(path, update=update) # download the vibe if it doesn't exist
        if vibe is not None:
            print(f'vibe already exists at {path}')
            return vibe
        print(f'vibe path: {path}')
        prompt = {
            'code': code,
            'tasks': [task, 'make sure the output_format follows the following within <OUTPUT_JSON> and </OUTPUT_JSON>' ],
            'gith path': self.git_path(module=module),
            'output_format': {
                            "vibe": "score out of 100", 
                            "dope_things_about_this": "list of dope things", 
                            "improvements": "list of improvements"
                            }
        }
        output = ''
        for ch in c.chat(prompt, process_input=False):
            print(ch, end='', flush=True)
            output += ch
        output = output.split('<OUTPUT_JSON>')[-1].split('</OUTPUT_JSON>')[0]

        return output

    def get_path(self, path):
        return c.abspath(f'~/.commune/vibe/{path}')

    def git_path(self, module='module', branch='main'):
        """
        Get the git path of the module.
        """
        dirpath =  c.dirpath(module)
        if dirpath.split('/')[-1] == c.repo_name:
            dirpath = os.path.dirname(dirpath)
        git_path = ''

        while len(dirpath.split('/')) > 0:
            git_config_path = dirpath + '/.git/config'
            if os.path.exists(git_config_path):
                git_url = c.get_text(git_config_path).split('url = ')[-1].strip().split('\n')[0]
                if git_url.endswith('.git'):
                    git_url = git_url[:-4]
                break
            else:
                git_path += dirpath.split('/')[-1]
                dirpath = os.path.dirname(dirpath)
        # get branch
        branch = c.get_text(dirpath + '/.git/HEAD').split('/')[-1].strip()
        return git_url +f'/tree/{branch}/' +git_path