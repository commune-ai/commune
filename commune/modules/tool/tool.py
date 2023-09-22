import commune as c

class Tool(c.Module):
    def __init__(
        self,
        # other params
        name: str = 'dam',
        description: str = 'This is a base tool that does nothing.',
        tags: list[str] = ['defi', 'tool'], 
        **kwargs
    ):
        self.set_config(kwargs=locals())
        
        self.name = name
        self.description = description
        self.tags = tags
        self.kwargs = kwargs

        ## DEFINE TOOL STUFF

    def call(self, x:int , y:int) -> int:
        return x * 2 + y
    
