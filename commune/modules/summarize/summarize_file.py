
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
              format = 'LIST(DICT(obj:str, desc:str))',
              max_age = 600,
              timeout= 30,
              update= False,
              verbose: bool = True) -> List[str]:

        assert os.path.exists(path), f"Path does not exist: {path}. Set force=True to ignore this error."
        cache_path = 'reuslts/' + path.replace('/', '_').replace('\\', '_')
        result = c.get(cache_path, max_age=max_age, update=update)
        if result is not None:
            print(f"Cache hit: {cache_path}")
            return result
        else:
            print(f"Cache miss: {cache_path}")
        
        if os.path.isdir(path):
            print(f"Processing directory: {path}")
            result = {}
            progress = c.tqdm(c.files(path), desc="Processing files")
            future2path = {}
            for p in c.files(path):
                params = {'path': p, 'query': query, 'model': model, 'temperature': temperature, 'task': task, 'format': format, 'max_age': max_age, 'update': update, 'verbose': verbose}
                future = c.submit(self.forward, params, timeout=timeout)
                future2path[future] = p
            for future in c.as_completed(future2path, timeout=timeout):
                progress.update(1)
                p = future2path[future]
                result[p] = future.result()
            return result   
        print(f"Processing file: {path}")

        # Format context if provided
        assert os.path.isfile(path), f"Path is not a file: {path}"
        content = c.text(path)

        prompt = f'''
        TASK
        - summarize the follwoing based on the format based on the wquery 
        - query --> {query}
        CONTENT
        {content} 
        RESULT_FORMAT
        {self.anchors[0]}({format}){self.anchors[1]}
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
        anchors = self.anchors
        output = anchors[0].join(output.split(anchors[0])[1:])
        output = anchors[1].join(output.split(anchors[1])[:-1])
        if verbose:
            print("\nParsing response...", color="cyan")
            
        result =   json.loads(output)
        c.put(cache_path, result)
    
        return result
