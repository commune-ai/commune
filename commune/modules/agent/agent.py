import commune as c

class Agent(c.Module):
    def __init__(self,
                name='agent',
                description='This is a base agent that does nothing.', 
                tags=['defi', 'agent'],
                llm = 'openai::gpt4',
                tools=[]
                ):
        self.name = name
        self.description = description
        self.tags = tags
        self.llm = llm
        self.tools = tools


    def call(self, prompt:str, memory: 'Memory' = None) -> str:
        return {
            'prompt': prompt,
            'response': 'This is a base agent that does nothing.',
            'history': []
            }

    # prompt tooling 


    @classmethod
    def find_tools(cls, prompt:str):
        raise NotImplementedError

    @classmethod
    def prompt2agent(cls, prompt:str) -> 'Agent':
        cls.find_tools(prompt, topk=5)

