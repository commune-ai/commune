
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print
class Summarize:
    """
    Advanced search and relevance ranking module powered by LLMs.
    
    This module helps find the most relevant items from a list of options based on a query,
    using LLM-based semantic understanding to rank and filter options.
    """
    anchors = ["<START_JSON>", "</END_JSON>"]
    def __init__(self, provider='dev.model.openrouter'):
        self.model = c.module(provider)()

    def get_cache_path(self, path):
        return  os.path.expanduser('~/.commune/summary/results/' + path)

    def forward(self,  
              path: str = __file__, # Path to the file containing options or a file  
              query: str = 'most relevant', 
              model: str = None,
              temperature: float = 0.5,
              task = None,
              update= False,
              verbose: bool = True) -> List[str]:

        assert os.path.exists(path), f"File not found: {path}"
        if os.path.isdir(path):
            result = {}
            for file in c.files(path):
                result[file] = self.forward(path=path, query=query, model=model, temperature=temperature, task=task, verbose=verbose)
            return result 
                    
        # Format context if provided
        assert os.path.isfile(path), f"Path is not a file: {path}"


        content = c.text(path)
        cache_path = + c.hash(path)

        cid = c.hash(path + query + str(model))
        cache_path = self.get_cache_path(cid)

        result = c.get(cache_path, None, update=update)


        # Build the prompt
        prompt = {
            "query": query,
            "content": content,
            "task": task if task else "summarize",
            "result_format": f"{self.anchors[0]}(LIST(DICT(obj:str, desc:str))){self.anchors[1]}"
        }
        prompt = json.dumps(prompt, indent=2)


        
        # Generate the response
        output = ''
        response = self.model.forward( 
            prompt, 
            model=model, 
            stream=True,
            temperature=temperature
        )

        # PROCEESS THE REPSONSE 
        for ch in response: 
            if verbose:
                print(ch, end='')
            output += ch

        output = self.anchors[0].join(output.split(self.anchors[0])[1:])
        output = self.anchors[1].join(output.split(self.anchors[1])[:-1])
        if verbose:
            print("\nParsing response...", color="cyan")
            
        result =   json.loads(output)

        c.put(cache_path, result)
    
        return result
