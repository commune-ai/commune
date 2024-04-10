import commune as c
from typing import *
import json

class ValiTextMMLU(c.module('vali')):

    def __init__(self,**kwargs):
        self.init_vali(locals())
        self.dataset = c.module('data.hf')(path=self.config.dataset)

    def score_module(self, module='model', **kwargs) -> int:


        model = c.connect(module) if isinstance(module, str) else module
        # get sample
        sample = self.dataset.sample()



        target = sample.pop(self.config.target) # a list of correct answers
        prompt = json.dumps({
        'sample': sample,
        'instruction': 'GIVE THE ANSWER AS AN INDEX -> {{answer:str}} ?',
        })
        output: str = model.generate(prompt)
        # process the output
        if isinstance(output, dict):
            output = output['answer']
        else:
            try:
                output = json.loads(output)['answer']
            except:
                pass
        # get the correct answer
        w = float(target in output)
        return {'w': w, 'target': target, 'input': prompt, 'output': output}


