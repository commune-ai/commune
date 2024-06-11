import commune as c

class ToolRegistry(c.Module):
    def __init__(
        self,
        tools = [],
    ):
        self.tool_map = {tool.name:tool for tool in tools}
    def call(self,tool:str , *args, **kwargs):
        self.tool_map[tool].forward(*args, **kwargs)
