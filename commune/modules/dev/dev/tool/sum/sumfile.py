
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print
class SumFile:
    """
    Advanced search and relevance ranking module powered by LLMs.
    
    This module helps find the most relevant items from a list of options based on a query,
    using LLM-based semantic understanding to rank and filter options.
    """

    task = """
        - summarize the follwoing based on the format based on the wquery 
        - if a function is an object of a class, then include the object name as class/function name
        - if a function is a method of a class, then include the class name as well
        - if a function is a module, then include the module name as well
        """
    anchors = ["<START_JSON>", "</END_JSON>"]
    result_format = f'{anchors[0]}(LIST(DICT(obj:str, desc:str))){anchors[1]}'
    cache_dir: str = '~/.summarize/cache'

    def __init__(self, model='model.openrouter'):
        self.model = c.module(model)()

    def abspath(self, path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))


    def forward(self,  
              path: str = __file__, # Path to the file containing options or a file  
              query: str = 'most relevant', 
              model: str = None,
              temperature: float = 0.5,
              content=  None,
              update = False,
              **kwargs) -> List[str]:

        path = self.abspath(path)        
        assert os.path.exists(path), f"File not found: {path}"
        if not os.path.isfile(path):
            return self.summarize_folder(path=path, query=query, model=model,   
                                         temperature=temperature, content=content, update=update, **kwargs)
        assert os.path.isfile(path), f"Path is not a file: {path}"
        print(f"Summarizing file: {path}", color="cyan")       
        content = content or  c.text(path)
        # hash
        prompt = f'''
        TASK={self.task}
        QUERY={query}
        CONTENT={content} 
        RESULT_FORMAT={self.result_format}
        '''
        path = self.cache_dir +  c.hash(prompt)
        result = c.get(path, update=update)
        # Generate the response
        if result != None:
            return result
        result = self.model.forward( prompt, model=model,  stream=True, temperature=temperature )
        return self.process_result(result, path=path)


    def process_result(self, response: Union[str, List[str]], path=None ) -> Any:
        """
        Process the response from the model, extracting the relevant JSON data.
        """
        output = ''
        for ch in response: 
            print(ch, end='')
            output += ch
        output = self.anchors[0].join(output.split(self.anchors[0])[1:])
        output = self.anchors[1].join(output.split(self.anchors[1])[:-1])
        result =   json.loads(output)
        if path:
            c.put(path, result)

        return result



    def summarize_folder(self, path: str = './', **kwargs) -> List[str]:
        """
        Summarize the contents of a folder.
        """
        files = c.files(path)
        results = {}
        n = len(files)
        for i, file in enumerate(files):
            print(f"Summarizing file {i + 1}/{n}: {file}")
            results[file] = self.forward(path=file, **kwargs)
        return results