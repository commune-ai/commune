import commune as c

class Agent(c.Module):
    def __init__(self,
                name='agent',
                description='This is a base agent that does nothing.', 
                tags=['defi', 'agent'],
                model = 'model.openai::obama12',
                tools=[]
                ):
        self.name = name
        self.description = description
        self.tags = tags
        self.model = c.connect(model)
        self.tools = tools

    def 


    def call(self, prompt:str, model=None) -> str:
        if model != None:
            self.model = c.connect(model)
        return self.model.generate(prompt)

    # prompt tooling 


    @classmethod
    def find_tools(cls, prompt:str):
        raise NotImplementedError

    @classmethod
    def prompt2agent(cls, prompt:str) -> 'Agent':
        cls.find_tools(prompt, topk=5)

