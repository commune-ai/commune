
import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print
class Tool:

    task = """
        - replace the content json with a new json based on the error message 
        - do this until it is properly loaded using json.loads
        - make sure the content is the same inside the data field
        - the purpose is to fix the json content while not changing the data
        - make sure to not return anything other than what the output format specifiees
        - DO NOT RETURN EMPTY FIELDS IF IT DOESNT MATCH THE OUTPUT FORMAT
        """
    anchors = ["<START_JSON>", "</END_JSON>"]
    output_format = 'DICT(data:str)'
    cache_dir: str = '~/.fix/cache'

    def __init__(self, model='anthropic/claude-opus-4', ):
        self.model = c.mod('model.openrouter')(model=model)
    def forward(self,  
              content: str = '{}',  # Path to the file containing options or a file
              history = [],
              trials = 4,
            
              **kwargs) -> List[str]:
        """
        Generate a valid JSON based on the error message and query.
        """
        for t in range(trials):
            prompt = f'''
                TASK={self.task}
                CONTENT={content} 
                OUTPUT_FORMAT={self.anchors[0]}{self.output_format}{self.anchors[1]}
                HISTORY={history}
                '''
            try:
                result = self.model.forward( prompt, stream=True)
                result = self.process(result)
                return  json.loads(result['data'])
            except json.JSONDecodeError as e:
                history.append(str(e))
        raise ValueError("Failed to process JSON after multiple attempts.")

    def process(self, result) -> Any:
        output = ''
        for ch in result: 
            print(ch, end='')
            output += ch

        # Trim based on anchors
        start = output.find(self.anchors[0])
        end = output.find(self.anchors[1], start)
        if start == -1 or end == -1:
            raise ValueError("Anchors not found in output.")

        json_str = output[start + len(self.anchors[0]): end]
        
        # Return the dict with the stringified json under 'data'
        return json.loads(json_str)

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
