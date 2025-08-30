
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

    
    def __init__(self, provider='dev.model.openrouter'):
        """
        Initialize the Find module.
        
        Args:
            model: Pre-initialized model instance (optional)
            default_provider: Provider to use if no model is provided
            default_model: Default model to use for ranking
        """
        self.model = c.module(provider)()
        self.anchors = ["<START_JSON>", "</END_JSON>"]

    def forward(self,  
              path: str = __file__, # Path to the file containing options or a file  
              query: str = 'most relevant', 
              model: str = None,
              temperature: float = 0.5,
              task = None,
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
        # hash
        cache_path = 'reuslts/' + c.hash(path)

        # Build the prompt

        prompt = f'''
        TASK
        - summarize the follwoing based on the format based on the wquery 
        - query --> {query}
        CONTENT
        {content} 
        RESULT_FORMAT
        {self.anchors[0]}(LIST(DICT(obj:str, desc:str))){self.anchors[1]}
        '''
        
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

        output = anchors[0].join(output.split(anchors[0])[1:])
        output = anchors[1].join(output.split(anchors[1])[:-1])
        if verbose:
            print("\nParsing response...", color="cyan")
            
        result =   json.loads(output)
    
        return result
