 import commune as c
import json




class Agent(c.Module):
    tools = ['module.cmd']

    description = """
    Use the tools to solve the problem. 
    USE MEMORY TO SOLVE THE PROBLEM
    store it in any way you like in the history,
    and use it in the next step
    IF YOU NEED TO USE A TOOL, USE THE TOOL TO SOLVE THE PROBLEM
    please file in the 
    WHEN YOU ARE DONE, PLEASE FILE IN THE ANSWER, AND FINISH THE PROMPT
    PLEASE SCORE YOUR CONFIDENCE IN THE ANSWER FROM 0 TO 1, 1 BEING THE MOST CONFIDENT

    """


    prompt = {
        'description': description,
        'prompt': 'This is the prompt',
        'tools': tools,
        'history': 'USE MEMORY TO SOLVE THE PROBLEM IN KNOWLEDGE TUPLES (HEAD, RELEATION, PAIR)(dict)',
        'thoughts': 'WRITE YOUR THOUGHTS HERE IN KNOWLEDGE TUPLES (dict)',
        'quit': 'INCLUDE THE FINISHED PROMPT HERE (bool)',
        'answer': 'INCLUDE THE ANSWER HERE',
        'confidence': 'SCORE YOUR CONFIDENCE IN THE ANSWER FROM 0 TO 1',
    }

    def __init__(self,
                name='agent',
                description : str = None, 
                model : str = 'model.openai',
                network : str = 'local',
                tools:list = tools
                ):
        self.name = name
        self.description = description if description != None else self.description
        self.set_model(model, network=network)
        self.set_tools(tools)


    def set_model(self, model:str = 'model.openai ', network:str = 'local'):
        self.model_namespace = c.namespace(search=model, netowrk=network)
        assert len(self.model_namespace) > 0, f"no models found in {model}, please check the model path"
        self.model_addresses = list(self.model_namespace.values())
        self.model_names = list(self.model_namespace.keys())
        self.network = network
        self.model = c.connect(c.choice(self.model_addresses))
        return {"success": True, "message": f"set model to {self.model}"}
    
    
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
    


    

    def call(self, 
             text:str,
             model=None, 
             history=None, 
             tools=tools, 
             n = 1,
             description:str = None) -> str:
        


        if model != None:
            self.model = c.connect(model)
        tools = self.resolve_tools(tools)
        history = history or []
        description = self.description if description == None else description

        for i in range(n):
            prompt = {
                'step': i,
                'max_steps': n, 
                'description': description,
                'input': text,
                'history': history,
                'tools': tools,
                'confidence': 0,
                'call_tool': {'tool': None, 'kwargs': None},
                'answer': None
            }
            output = self.model.generate(json.dumps(prompt), max_tokens=512)
            if 'data' in output:
                output = output['data']
            output = json.loads(output)
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
        tool_info = module.fn_schema(fn, docs=True)
        return tool_info

    
    def get_tools(self, tools:list, fn_seperator:str = '.'):
        return {t: self.get_tool(t, fn_seperator=fn_seperator) for t in tools}
    
    def add_tool(self, tool:str):
        schema = self.schema(tool)
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
    
