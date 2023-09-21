import commune as c

class Tool(c.Module):
    def __init__(
        self,

        # other params
        name: str,
        description: str = 'This is a base tool that does nothing.',
        tags: list[str] = ['defi', 'tool'],
    ):
        
        self.name = name
        self.description = description
        self.tags = tags

    def forward(self, x:int) -> int:
        return x * 2
    
