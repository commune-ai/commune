
import commune as c
import os
from typing import Dict, Any, Optional, Union
from .utils import abspath, put_text, ensure_directory_exists

class CreateFile:
    """
    A utility tool for creating new files at specified paths.
    
    This class provides functionality to:
    - Create new files with specified content
    - Ensure parent directories exist
    - Handle different file types appropriately
    - Provide feedback on the operation
    """
    
    def __init__(self, **kwargs):
        self.model = c.mod('model.openrouter')() 
           
    def forward(self, 
                path: str=os.path.dirname(__file__), 
                query = "make a docker container",
                mod = None,
                overwrite: bool = False,
                verbose: bool = True) -> Dict[str, Any]:

        print(f"Creating file at {path} with query: {query}")
        if mod is not None:
            path = c.dirpath(mod)
        context = c.fn('select.files/')(path)
        prompt = {
            'query': query,
            'system': "create a docker image based on the following context",
            'context': context,
            'result_format': 'output between <START_JSON>(dockerfile:str)<END_JSON> tags for JSON output',
        }
        # Check if file already exists
        content = ''

        response = self.model.forward(str(prompt), stream=1)
        for chunk in response:
            print(chunk, end='', flush=True)
            content += chunk

        path = path + '/Dockerfile'
        return {
            'path': path,
            'content': content,
            'success': True,
            'message': f"File created at {path}",
        }