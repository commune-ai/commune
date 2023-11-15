import commune as c
import json




class Agent(c.Module):
    description = """You have a set of tools, and you should call them if you need to 
    call_tools: {tool:str, kwargs:dict} please fill in the tool, args, and kwargs
    only return the response and call_tool field
    """
    def __init__(self,
                name='agent',
                description= None, 
                model = 'model.openai',
                tools:list = []
                ):
        self.name = name
        self.description = description if description != None else self.description
        self.set_model(model)
        self.set_tools(tools)


    def set_model(self, model:str):
        self.model = c.connect(model)
        return self.model

    
    def rm_tools(self, tools:list = None):
        if tools == None:
            self.tools = {}
        else:
            for t in tools:
                self.rm_tool(t)
        return self.tools
    

    def resolve_tools(self, tools):

        if isinstance(tools, str):
            tools = [tools]
        if isinstance(tools, list):
            tools = self.get_tools(tools)
        if tools == None:
            tools = self.tools

        return tools
    
    default_tools = ['module.ls', 'module.fns', 'module.servers', 'module.modules', 'module.module']

    def call(self, prompt:str, model=None, history=None, tools=default_tools, description:str = None) -> str:
        if model != None:
            self.model = c.connect(model)
        tools = self.resolve_tools(tools)
        if description == None:
            description = self.description

        prompt = {
            'description': description,
            'prompt': prompt,
            'history': history,
            'tools': tools,
            'purpose': ['Please think about it, you have a set of tools, and you should call them if you need to', 
                            'call_tools: {tool:str, kwargs:dict} please fill in the tool, args, and kwargs', 
                            'briely say why you are using this tool',
                            'feel free to write thoughts in the thoughts field', 
                            'RETURN THE  FOLLOWING FIELDS: response, call_tool, thoughts, confidence'
                            ],
            'response': None,
            'thoughts': None,
            'confidence': None,
            'call_tool': {'tool': None, 'kwargs': None}
        }
        output = self.model.generate(json.dumps(prompt))
        c.print(output)
        output = json.loads(output)
        prompt.update(output)
        return output
    # prompt tooling 
    generate = call 


    @classmethod
    def find_tools(cls, prompt:str):
        raise NotImplementedError

    @classmethod
    def prompt2agent(cls, prompt:str) -> 'Agent':
        cls.find_tools(prompt, topk=5)




    


    def set_tools(self, tools:list):
        self.tools = {}
        self.add_tools(tools)
        return self.tools
    
    def add_tools(self, tools:list):
        for t in tools:
            self.add_tool(t)
        return self.tools
    
    def get_tool(self, tool:str, fn_seperator:str = '.'):
        module = fn_seperator.join(tool.split(fn_seperator)[:1])
        fn = tool.split(fn_seperator)[1]
        module = c.module(module)
        schema = module.schema(docs=True)
        return schema[fn]
    
    def get_tools(self, tools:list, fn_seperator:str = '.'):
        return {t: self.get_tool(t, fn_seperator=fn_seperator) for t in tools}
    
    def add_tool(self, tool:str, fn_seperator:str = '.'):
        schema = self.get_tool_schema(tool, fn_seperator=fn_seperator)
        self.tools[tool] = schema
        return self.tools
    
    def rm_tool(self, tool:str):
        del self.tools[tool]
        return self.tools
    


    def test_model(self, prompt:str, model=None, history=None, **kwargs):
        if model != None:
            self.model = c.connect(model)

        prompt = {
            'description': self.description,
            'prompt': prompt,
            'history': history,
            'response': None,
            'instruction': 'complete response'
        }

        output = self.model.generate(json.dumps(prompt))

        prompt.update(json.loads(self.model.generate(json.dumps(prompt))))
        return prompt
    
    def test(self, prompt:str='hey', model=None, history=None, **kwargs):
        response =  self.call(prompt, model=model, history=history, **kwargs)

        assert 'response' in response, f"response not in {response}"
        assert isinstance(response['response'], str), f"response is not a string: {response['response']}"
        return {
            'prompt': prompt,
            'response': response['response'],
            'success': True,
            }
    
