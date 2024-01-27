import commune as c
import json

class Coder(c.Module):
    def call(self,
             fn='coder/call', 
             model = 'model.openai',
             timeout=10,
             **model_params):
        '''
        ## Function Documentation
        
        ### `call` method
        
        **Description:**
        
        The `call` method is responsible for connecting to a specified model, sending a code input along with an instruction to document the function in a professional manner, and then retrieving the generated documentation.
        
        **Parameters:**
        
        - `fn` (str): The filename or identifier of the code function to document. Defaults to `'coder/call'`.
        - `model` (str): The model identifier to use for generating the documentation. Defaults to `'model.openai'`.
        - `timeout` (int): The maximum amount of time (in seconds) to wait for the model to generate the documentation. Defaults to `10`.
        - `**model_params`: Additional parameters that are passed to the model upon connection.
        
        **Returns:**
        
        - `docs` (str): The generated documentation for the given function.
        
        **Usage Example:**
        
        ```python
        # Instantiate the class
        c = SomeClass()
        
        # Call the 'call' method to generate documentation
        documentation = c.call(
            fn='my_function',
            model='model.openai',
            timeout=15,
            param1=value1,
            param2=value2
        )
        
        # 'documentation' variable now contains the generated documentation for 'my_function'
        ```
        
        **Notes:**
        
        - The method generates the documentation by first connecting to the model with the provided model parameters.
        - It then sends a structured input containing the instruction and code to the model.
        - The model is expected to generate documentation based on this input within the specified timeout period.
        - The generated documentation is then processed and added to the function identified by `fn`.
        - The method returns the processed documentation for further use or inspection.
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
    
    comment = call
    document_fn = call

    
    
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

