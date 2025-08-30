

import os
import json
import commune as c



class Score:
    def __init__(self, agent='agent'):
        self.agent = c.module(agent)()


    def get_code(self, module:str):
        if '.git' in module:
            git_url = module
            tmp_dir = '/tmp/score'
            cmd = f'git clone {git_url} {tmp_dir}'
            c.cmd(cmd, verbose=1)
            code = c.file2text(tmp_dir)
            # deleting the .git folder to avoid any issues with git
            c.cmd(f'rm -rf {tmp_dir}/.git')
        elif c.path_exists(module):
            code = c.file2text(module)
        elif c.module_exists(module):
            code = c.code_map(module)
        return code
    def forward(self, module:str, **kwargs):
        code = self.get_code(module)
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
        return json.loads(output.split('<OUTPUT>')[1].split('</OUTPUT>')[0])
