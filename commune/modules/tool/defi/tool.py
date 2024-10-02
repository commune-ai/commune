import commune as c

class Tool(a.Block):
    name = None
    description = 'This is a base tool that does nothing.'
    tags = ['defi', 'tool']

    def __init__(
        self,
        **kwargs
    ):
        ## DEFINE TOOL STUFF
        pass

    def call(self, x:int = 1, y:int= 1) -> int:
        return x * 2 + y 
    
    @classmethod
    def tool_list(cls):
        import random 
        tool2info =  a.tool2info()
        tool_list = []
        for tool, tool_info in tool2info.items():
            print(tool_info['name'])
            tool_list = tool_list + [tool_info['name']]
            tool_list[-1]['price'] = random.randint(1, 100)
        
        return tool_list
    
    @classmethod
    def info(cls):
        return {
            'name': cls.name if cls.name else cls.block_name(),
            'description': cls.description,
            'tags': cls.tags,
            'schema': cls.get_schema('call'),
        }
    
    @classmethod
    def tool2info(self):
        for tool in a.tools(info=True):
            print(tool.info())
    
    @classmethod
    def tool2info(cls):
        tools = a.tools()
        tool2info = {}
        for tool in tools:
            tool_info = a.block(tool).info()
            tool2info[tool_info['name']] = tool_info
        return tool2info
    

    @classmethod
    def filepath(cls):
        import inspect
        return inspect.getfile(cls)
    
    @classmethod
    def code(cls):
        return cls.get_text(cls.filepath())
    

    @classmethod
    def get_general_schema(cls):
        get_general_schema = a.import_object('autonomy.tool.openai_helper.get_general_schema')
        return get_general_schema(cls.fncode('call'))
    
    

    
    
