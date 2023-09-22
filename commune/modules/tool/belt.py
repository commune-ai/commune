import commune as c

class ToolBelt(c.Module):
    def __init__(
        self,
        tools = [],
    ):
        self.tool_map = {tool.name:tool for tool in tools}
    def forward(self,tool:str , *args, **kwargs):
        self.tool_map[tool].forward(*args, **kwargs)