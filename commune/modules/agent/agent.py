import commune as c
import json

class Agent(c.Module):
    def __init__(self,
                name='agent',
                description='This is a base agent that does nothing.', 
                tags=['defi', 'agent'],
                model = 'model.openai',
                tools=[]
                ):
        self.name = name
        self.description = description
        self.tags = tags
        self.model = c.connect(model)
        self.tools = tools




    def call(self, prompt:str, model=None, history=None,) -> str:
        if model != None:
            self.model = c.connect(model)

        prompt = {
            'description': self.description,
            'prompt': prompt,
            'history': history,
            'response': None,
            'instruction': 'complete response'
        }

        prompt.update(json.loads(self.model.generate(json.dumps(prompt))))
        return prompt
    # prompt tooling 


    @classmethod
    def find_tools(cls, prompt:str):
        raise NotImplementedError

    @classmethod
    def prompt2agent(cls, prompt:str) -> 'Agent':
        cls.find_tools(prompt, topk=5)

