import commune as c
import json
from typing import *

class Coder(c.Module):
    def comment(self,
             fn='coder/call', 
             model = 'model.openai',
             timeout=20,
             **model_params):
        '''
        ### Function Documentation
        
        #### `call(self, fn='coder/call', model='model.openai', timeout=20, **model_params)`
        
        This function is responsible for generating documentation for a given piece of code by utilizing a language model. 
        
        Parameters:
        - `fn` (str): The name of the function that needs documentation. Default value is `'coder/call'`.
        - `model` (str): The identifier of the language model to be used. Default is `'model.openai'`.
        - `timeout` (int): The maximum amount of time (in seconds) to wait for the model to generate the documentation. Default is `20`.
        - `**model_params`: Arbitrary keyword arguments that will be passed to the `connect` method of the `c` object when connecting to the language model.
        
        Returns:
        - `docs` (str): The generated documentation for the specified code.
        
        The function performs the following steps:
        1. Connects to the specified language model using the provided parameters.
        2. Constructs an input JSON object containing the instruction, code, and a placeholder for documentation.
        3. Requests the language model to generate documentation based on the provided input.
        4. Processes the generated documentation response.
        5. Adds the generated documentation to the function using the `c.add_docs()` method.
        6. Returns the generated documentation.
        
        **Example Usage:**
        
        ```python
        # assuming the 'c' object and 'call' method are part of a class
        caller = YourClass()
        documentation = caller.call(
            fn='your_function_name',
            model='your_model_identifier',
            timeout=30,
            model_params={'additional': 'parameters'}
        )
        print(documentation)
        ``` 
        
        **Note:** 
        - The `c` object is assumed to be a pre-defined object with methods `connect`, `fn_code`, and `add_docs`.
        - `self.process_response` is assumed to be a method that processes the generated documentation response. Its functionality is not detailed in the provided code.
        '''
        model = c.connect(model, **model_params)
        input = json.dumps({
            'instruction': 'given the code, document the function in a professional manner in the docs section', 
            'code': c.fn_code(fn),
            'docs': None,
        })
        # get the docs
        docs = model.generate(input, timeout=timeout)
        docs = self.process_response(docs)

        # add docs to the function
        c.add_docs(fn, docs)

        return docs
    
    call = document_fn = comment
    
    def document_module(self,
             module='agent.coder', 
             fns = None,
             model = 'model.openai',
             **model_params
             ):
        fns = c.module(module).fns()
        for fn in fns:
            c.print(f'Documenting function {fn} in module {module}...')

            try:
                future = c.submit(self.document_fn, dict(fn=module+'/'+fn, model=model, **model_params))
                future.result()
            except:
                c.print(f'Failed to document function {fn} in module {module}...')
            print(f'Documenting function {fn} in module {module}...')

        return 
    
    def process_response(self, response):
        '''
        """
        Documentation for `process_response` function:
        
        This function is responsible for processing a given response and ensuring it's in a proper JSON format. If the response is in a string format, the function attempts to load it as a JSON object. If the loading fails, it simply passes without raising any exceptions.
        
        Parameters:
            - self: The instance of the class that this method is bound to.
            - response: A response object that is to be processed. It can be a string or already a
        '''
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except:
                pass
        
        return response



    def file2fns(self, filepath):
        '''
        """
        Documentation for `get_fns` function:
        
        This function retrieves the list of functions available in a given module.
        
        Parameters:
            - self: The instance of the class that this method is bound to.
            - module: The name of the module for which the list of functions is to be retrieved.
        
        Returns:
            - fns: A list of function names available in the specified module.
        '''

        if c.module_exists(filepath):
            filepath = c.filepath()
        if not filepath.endswith('.py'):
            filepath = filepath + '.py'
        code =  c.get_text(filepath)
        lines = code.split('\n')
        fns = []
        for line in lines:
            if  '):' in line.strip() and 'def ' in line.split('):')[0].strip():
                fn = line.split('def ')[1].split('):')[0].split('(')[0]
                if ' ' in fn or ']' in fn:
                    continue
                fns.append(fn)
                

        return fns
    

    @property
    def get_function_default_map(self, include_parents=False):
        return self.get_function_default_map(obj=self, include_parents=False)
        
    @classmethod
    def get_function_default_map(cls, obj:Any= None, include_parents=False) -> Dict[str, Dict[str, Any]]:
        obj = obj if obj else cls
        default_value_map = {}
        function_signature = cls.fn_signature_map(obj=obj,include_parents=include_parents)
        for fn_name, fn in function_signature.items():
            default_value_map[fn_name] = {}
            if fn_name in ['self', 'cls']:
                continue
            for var_name, var in fn.items():
                if len(var.split('=')) == 1:
                    var_type = var
                    default_value_map[fn_name][var_name] = 'NA'
 
                elif len(var.split('=')) == 2:
                    var_value = var.split('=')[-1].strip()                    
                    default_value_map[fn_name][var_name] = eval(var_value)
        
        return default_value_map   
    


    def file2file(self, path, **kwargs):
        '''
        """
        Documentation for `file2file` function:
        
        This function reads the content of a file and writes it to another file.
        
        Parameters:
            - self: The instance of the class that this method is bound to.
            - path: The path to the file to be read.
            - new_path: The path to the file to be written. If not provided, the content is written to the same file.
        
        Returns:
            - success: A boolean value indicating whether the operation was successful.
        '''
        content = c.get_text(path)
        content = self.model.forward(content, **kwargs)
        c.put_text(path, content)
        return content
    