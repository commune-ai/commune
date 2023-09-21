import commune as c

class Agent(c.Module):
    def __init__(self,
                name='agent',
                description='This is a base agent that does nothing.', 
                tags=['defi', 'agent'],
                llm = 'openai::gpt4',
                tools=[],):
        self.name = name
        self.description = description
        self.tags = tags
        self.llm = llm
        self.tools = c.module('tool.belt')(tools=tools)