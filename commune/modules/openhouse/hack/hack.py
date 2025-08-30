import commune as c
import os
import json
import getpass
class Hackathon:

    dev = c.module('dev')()
    key_manager = c.module('eth.key')()
    mods_path = c.abspath('./hack/modules')
    
    def password2key(self, password:str=None):
        """
        convert a password to a key
        """
        if password is None:
            password = getpass.getpass('Enter password: ')
            repeat_password = getpass.getpass('Repeat password: ')
            if password != repeat_password:
                raise ValueError('passwords do not match')
        key = self.key_manager.from_password(password)
        return key
    def submit(self, query=None, name=None, password:str=None):
        
        name = input('Enter module name: ') if name is None else name
        assert not self.score_exists(name), f'module {name} already exists'
        query = input('Enter query: ') if query is None else query
        key = self.password2key(password)
        module_path = self.mods_path + '/' + name
        output =  self.dev.forward(*query, target=module_path, path=None, force_save=True)
        return  self.score(name, key=key)

    def get_module_path(self, module:str):
        return self.mods_path + '/' + module
    def str2key(self , key:str):
        return self.key_manager.from_password(key)
    # def create_key():

    def modules(self):
        """
        list all the modules in the modules path
        """
        return [p.split('/')[-1] for p in c.ls(self.mods_path)]

    def get_module_code(self, module:str):
        """
        get the code of a module
        """
        module_path = self.mods_path + '/' + module
        if not os.path.exists(module_path):
            raise ValueError(f'module {module} does not exist')
        return c.file2text(module_path)

    def score(self, module, key, **kwargs):
        self.agent = c.module('agent')()
        
        code = self.get_module_code(module)
        prompt = f"""
        GOAL:
        score the code out of 100 and provide feedback for improvements 
        and suggest point additions if they are included to
        be very strict and suggest points per improvement that 
        you suggest in the feedback
        YOUR SCORING SHOULD CONSIDER THE FOLLOWING VIBES:
        - readability
        - efficiency
        - style
        - correctness
        - comments should be lightlu considered only when it makes snes, we want to avoid over commenting
        - code should be self explanatory
        - code should be well structured
        - code should be well documented
        CODE: 
        {code}
        OUTPUT_FORMAT:
        <OUTPUT>DICT(score:int, feedback:str, suggestions=List[dict(improvement:str, delta:int)]])</OUTPUT>
        """
        output = ''
        for ch in  self.agent.forward(prompt, stream=True, **kwargs):
            output += ch
            print(ch, end='')
            if '</OUTPUT>' in output:
                break
        score_output =  json.loads(output.split('<OUTPUT>')[1].split('</OUTPUT>')[0])
        score_path = self.mods_path + '/' + module  + '/score.json'
        key_address = key.address
        score_output['key'] = key_address
        c.put_json(score_path, score_output)
        assert os.path.exists(score_path), f'score file {score_path} does not exist'
        return score_output

    def get_score_path(self, module:str):
        """
        get the score path of a module
        """
        module_path = self.mods_path + '/' + module
        return module_path + '/score.json'

    def score_exists(self, module:str):
        """
        check if the score path of a module exists
        """
        score_path = self.get_score_path(module)
        return os.path.exists(score_path)


    def score_paths(self):
        """
        g,et the score paths of all modules
        """
        return [self.get_score_path(module) for module in self.modules()]

    def leaderboard(self, features=['score', 'key', 'module']):
        """
        get the leaderboard of all modules
        """
        leaderboard = []
        for module in self.modules():
            score_path = self.get_score_path(module)
            if os.path.exists(score_path):
                score = c.get_json(score_path)
                score['module'] = os.path.dirname(score_path).split('/')[-1]
                leaderboard.append(score)
        df =  c.df(leaderboard)
        return df

    def verify(self, module:str, password:str=None):
        """
        verify the ownership of a module
        """
        score_path = self.get_score_path(module)
        score_data = c.get_json(score_path)
        assert 'key' in score_data, f'score file {score_path} does not contain a key'
        key = self.password2key(password)
        return key.address == score_data['key']
           