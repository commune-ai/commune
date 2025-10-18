
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print
class Tool:

    task = """
        generate a valid json based on the error message
        based on the query
        """
    anchors = ["<START_JSON>", "</END_JSON>"]

    def __init__(self, provider ='model.openrouter'):
        self.model = c.mod(model)()

    def forward(self,  
              query = 'anything maximally imaginative',
              history = [],
              trials = 4,
              output_format: str = 'DICT(data:str)',
              **kwargs) -> List[str]:
        # hash
        try:
            result = json.loads(content)
            error_msg = ''
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON content: {e}"

        for t in range(trials):
            prompt = f'''
                TASK={self.task}
                CONTENT={content} 
                QUERY={query}
                OUTPUT_FORMAT={self.anchors[0]}{self.output_format}{self.anchors[1]}
                ERROR_MSG={error_msg}
                HISTORY={history}
                '''
            try:
                result = self.model.forward( prompt, model=model,  stream=True)
                result = self.process(result)
                return  json.loads(result['data'])
            except json.JSONDecodeError as e:
                history.append(str(e))
        raise ValueError("Failed to process JSON after multiple attempts.")



    def process(self, result) -> Any:
        """
        Process the response from the model, extracting the relevant JSON data.
        """
        output = ''
        for ch in result: 
            print(ch, end='')
            output += ch
        output = self.anchors[0].join(output.split(self.anchors[0])[1:])
        output = self.anchors[1].join(output.split(self.anchors[1])[:-1])
        result =   json.loads(output)
        # verify if the result is a list of dicts
        return result

    def test(self, content: str = {'fam': 1, 'daweg': ['fam', 1]}, query: str = 'most relevant', model: str = None, **kwargs) -> List[str]:
        """
        Test the model with a given content and query.
        
        Args:
            content: JSON content to process
            query: Query to ask the model
            model: Model to use for processing
            **kwargs: Additional arguments
            
        Returns:
            Processed result from the model
        """
        content = json.dumps(content)
        disturbed_content = content + '}}'  # Disturb the content to trigger error
        new_json =  self.forward(content=disturbed_content, query=query, model=model, **kwargs)
        assert isinstance(new_json, dict), f"Expected a dictionary but got {type(new_json)}"
        # compare new and old json 
        old_json = json.loads(content)
        assert old_json == new_json, f"Old JSON: {old_json}, New JSON: {new_json}"
        return { "success": True, "data": new_json , "message": "JSON processed successfully" }
