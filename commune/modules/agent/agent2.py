import commune as c
import json

class Agent(c.Module):
    output_schema = """
    json '''
    {
        "response": "str",
        "call_tool": {
            "tool": "str",
            "kwargs": "dict",
        }
    }'''
    """

    description = """
    to call a tool {tool} with kwargs {kwargs}, 
    SDuse the following format: {tool: 'module.fn', kwargs: {}}"""

    def __init__(self,
                name='agent',
                description : str = None, 
                model : str = 'model.openrouter',
                network : str = 'local',
                tools:list = ['module.cmd']
                ):
        
        self.name = name
        self.description = description if description != None else self.description
        self.set_model(model, network=network)
        self.set_tools(tools)

    def set_model(self, model:str = 'model.openai ', network:str = 'local'):
        self.model = c.module(model)()
        return {"success": True, "message": f"set model to {self.model}"}
    
    
    def rm_tools(self, tools:list = None):
        if tools == None:
            self.tools = {}
        else:
            for t in tools:
                self.rm_tool(t)
        return self.tools
    _tools = {}
    @property
    def tools(self):
        return self._tools
    @tools.setter
    def tools(self, tools):
        self._tools = tools
        return self.tools
    
    

    def resolve_tools(self, tools):

        if isinstance(tools, str):
            tools = [tools]
        if isinstance(tools, list):
            tools = self.get_tools(tools)
        if tools == None:
            tools = self.tools

        return tools
    

    def talk(self, *text:str, **kwargs):
        text = ' '.join(list(map(str, text)))
        return self.call(text, **kwargs)
    def call(self, 
             text:str,
             model=None, 
             history=None, 
             max_tokens:int = 1000,
             n = 1,
             description:str = None) -> str:
        

    

        if model != None:
            self.model = c.connect(model)
        history = history or []
        description = self.description if description == None else description

        for i in range(n):
            prompt = {
                'input': text,
                'tools': self.tools,
                'description': description,

            }
            output = self.model.generate(json.dumps(prompt), max_tokens=max_tokens)
            output = json.loads(output.replace("'", '"'))

            prompt.update(output)
            if 'call_tool' in output:
                tool = output['call_tool']['tool']
                kwargs = output['call_tool']['kwargs']
                if kwargs == None:
                    kwargs = {}
                if tool != None:
                    module = '.'.join(tool.split('.')[:-1])
                    fn = tool.split('.')[-1]
                    module = c.module(module)
                    fn_type = module.classify_fn(fn)
                    if fn_type == "self":
                        module = module()
                    try:
                        response = getattr(module, fn)(**kwargs)
                    except Exception as e:
                        response = c.detailed_error(e)
                    
                    
                    output['call_tool']['response'] = response
                    history.append(output['call_tool'])
        
        
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
        self.tools = self.add_tools(tools)
        return self.tools
    
    def add_tools(self, tools:list):
        for t in tools:
            self.add_tool(t)
        return self.tools
    
    def get_tool(self, tool:str, fn_seperator:str = '.'):
        module = fn_seperator.join(tool.split(fn_seperator)[:1])
        fn = tool.split(fn_seperator)[1]
        module = c.module(module)
        tool_info = module.fn_schema(fn, docs=True)
        return tool_info

    
    def get_tools(self, tools:list, fn_seperator:str = '.'):
        return {t: self.get_tool(t, fn_seperator=fn_seperator) for t in tools}
    
    def add_tool(self, tool:str):
        if '.' in tool:
            module, tool = tool.split('.')
        module = c.module(module)
        schema = module.schema(tool)['default']
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
            }
    
    
    def aistr2json(self, s:str):
        self.model.call("")


    def edit_suggestions(self, description, file=None):
        if file != None:
            with open(file, 'r') as f:
                prompt = f.read()

        prompt = '\n'.join([f'{i} {line}' for i, line in enumerate(prompt.split('\n'))])
        prompt = f"""
        ---DESCRIPTION----
        {description}
        ----TEXT---
        {prompt}

        REPLY IN JSON FORMAT
        EACH SUGGESTION IS A JSON OBJECT
        {
         'line': 'int', 
         'description': 'str', 
         'edit': 'str'
         }
        """

        return prompt
        
    
