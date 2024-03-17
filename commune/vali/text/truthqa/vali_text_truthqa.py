import commune as c
from typing import *
import json

Vali = c.module('vali')
class ValiTextTruthfulQA(Vali):
    def __init__(self,**kwargs):
        config = self.set_config(kwargs=kwargs)
        self.dataset = c.module(self.config.dataset)()
        self.init_vali(config)


    def create_prompt(self, sample: dict) -> str:
        # format the prompt
        prompt = f'''
        {sample}
        INSTRUCTIONS: ANSWER THE ABOVE MULTIPLE CHOICE?
        EXAMPLE: {{answer:0}}
        ANSWER_JSON:
        ```json'''
        return prompt

    def score_module(self, module='model', **kwargs) -> int:


        model = c.connect(module) if isinstance(module, str) else module
        # get sample
        sample = self.dataset.sample()

        target = sample.pop(self.config.target) # a list of correct answers
        # create the prompt
        prompt = self.create_prompt(sample)

        # generate the output
        output: str = model.generate(prompt)
        c.print(output)
        prediction = json.loads(output)['answer']



        

        # get the correct answer
        w = float(prediction in target)

        if w == 0:
            w = self.config.base_score

        return {'w': w, 'target': target, 'prediction': prediction, 'sample': sample, 'output': output}

            
    @classmethod
    def test(cls, module='model.openai', **kwargs):
        vali = cls(start=False)
        return vali.score_module(module=module, **kwargs)
