
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print
class SelectFiles:

    def __init__(self, provider='model.openrouter'):
        """
        Initialize the Find module.
        
        Args:
            model: Pre-initialized model instance (optional)
            default_provider: Provider to use if no model is provided
            default_model: Default model to use for ranking
        """
        self.model = c.module(provider)()

    def forward(self,  
              path: Union[List[str], Dict[Any, str]] = './',  
              query: str = 'most relevant', 
              n: int = 10, 
              mod=None,
              content: bool = True,
               **kwargs) -> List[str]:

        if mod:
            path = c.dirpath(mod)
        results = c.fn('dev.tool.select/')(
            query=query,
            options= c.files(path),
            n=n,
            **kwargs
        )
        results =  [os.path.expanduser(file) for file in results]
        if content:
            results = {f:self.get_text(f) for f in results}
        return results

    def get_text(self, path: str) -> str:
        path = os.path.abspath(os.path.expanduser(path))
        with open(path, 'r') as f:
            text = f.read()
        return text

    def test(self):
        dirpath = os.path.dirname(__file__)
        query = f'i want to find the {__file__} file'
        results = self.forward(path=dirpath, query=query, n=1)
        assert __file__ in results, f"Expected {dirpath} in results, got {results}"
        print(f"Test results: {results}", color="green")
        return {
            "success": True,
            "message": f"Tesst with query '{query}' and wanted file '{__file__}'"
            
        }

        



    