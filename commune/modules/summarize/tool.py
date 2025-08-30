
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any
import importlib
import inspect

print = c.print

class Tool:


    fn = 'forward'
    """
    A toolbox that provides access to various tools and can intelligently select
    the most appropriate tool based on a query.
    
    This module helps organize and access tools within the dev.tool namespace,
    with the ability to automatically select the most relevant tool for a given task.
    """
    def __init__(self, model='dev.model.openrouter', prefix='dev.tool'):

        self.prefix = prefix
        self.model = c.module(model)()

    def forward(
        self, 
        query: str = 'i want to edit a file of ./', 
        *extra_query,
        tools: Optional[List[str]] = None, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward the query to the appropriate tool based on the provided query.
        
        Args:
            query (str): The query to be processed.
            tools (List[str], optional): List of specific tools to consider. If None, all tools are considered.
            **kwargs: Additional arguments to pass to the selected tool.
        
        Returns:
            Dict[str, Any]: The result from the selected tool.
        """
        
        # Forward the query to the selected tool
        prompt = self.preprocess(" ".join([query] + list(extra_query)))
        output  =  self.model.forward(prompt, **kwargs)
        output = self.postprocess(output)
        return output

        
    def preprocess(self, prompt):

        selector = c.mod('dev.tool.select')()
        tool2schema  = self.tool2schema()
        module =  selector.forward(query=query, options=tool2schema, n=1, **kwargs)[0]['name']
        tool_schema = c.schema(module)
        # get the function name from the module
        fn_name = selector.forward(query=query, options=tool_schema, n=1, **kwargs)[0]['name']
        anchors = ['<PARAMS>', '</PARAMS>']
        prompt = str({
            "query": query,
            "schema": schema[fn_name],
            'task': 'your goal is to create a plan of selecting at least one tool to be execturd',
            'outputformat': f"{anchors[0]}\nLIST(DICT(FN, PARAMS))\n{anchors[1]}\n",
        })

    def postprocess(self, output):
        return json.loads(output.split(anchors[0])[1].split(anchors[1])[0])


    def tools(self) -> List[str]:
        return [t for t in  c.mods() if t.startswith(self.prefix)]


    def tool2code(self) -> str:
        tool2schema = {
            tool: c.schema(tool, include_code=True)
            for tool in self.tools()
        }
        
    
    def tool2schema(self) -> Dict[str, str]:
        """
        Map each tool to its schema.
        
        Returns:
            Dict[str, str]: Dictionary mapping tool names to their schemas.
        """
        tool2schema = {}
        for tool in self.tools():
            tool2schema[tool] = self.schema(tool)
            tool2schema[tool].pop('name', None)
            tool2schema[tool].pop('format', None)
        return tool2schema
    
    def schema(self, tool: str,) -> Dict[str, str]:
        """
        Get the schema for a specific tool.
        
        Args:
            tool (str): The name of the tool.
        
        Returns:
            Dict[str, str]: The schema for the specified tool.
        """
        fn = self.fn
        schema =  c.fn_schema(getattr( c.module(tool), fn))
        schema['input'].pop('self', None)
        params_format = ' '.join([f'<{k.upper()}>{v["type"]}</{k.upper()}>' for k,v in schema['input'].items()]) 
        fn_format = f'FN::{fn.upper()}'
        schema['format'] =  f'<{fn_format}>' + params_format + f'</{fn_format}>'
        return schema

    def tool2code(self) -> Dict[str, str]:
        """
        Map each tool to its code.
        
        Returns:
            Dict[str, str]: Dictionary mapping tool names to their code.
        """
        tool2code = {
            tool: c.code(tool)
            for tool in self.tools()
        }
        return tool2code

    def tool2size(self) -> Dict[str, int]:
        """
        Map each tool to its code size.
        
        Returns:
            Dict[str, int]: Dictionary mapping tool names to their code size.
        """
        tool2code_size = {
            tool: len(c.code(tool))
            for tool in self.tools()
        }
        return tool2code_size

