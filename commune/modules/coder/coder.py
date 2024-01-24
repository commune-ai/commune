import commune as c
import json

class Coder(c.Module):
    def call(self,
             fn='agent.coder/call', 
             model = 'model.openai',
             **model_params):
        '''
        ### Function Documentation
        
        #### Function Name: 
        `call`
        
        #### Parameters: 
        - `fn` (str): The function name to be documented. Default value is `'agent.coder/call'`.
        - `model` (str): The model name to connect to for generating the documentation. Default value is `'model.openai'`.
        - `**model_params`: Arbitrary keyword arguments to be used as parameters when connecting to the model.
        
        #### Description: 
        The `call` function is used to
        '''
        '''
        ### Function Documentation
        
        #### Function Name: 
        `call`
        
        #### Parameters: 
        - `fn` (str): The function name to be documented. Default value is `'agent.coder/call'`.
        - `model` (str): The model name to connect to for generating the documentation. Default value is `'model.openai'`.
        - `**model_params`: Arbitrary keyword arguments to be used as parameters when connecting to the model.
        
        #### Description: 
        This function is used to automatically generate documentation
        '''
        model = c.connect(model, **model_params)
        input = json.dumps({
            'instruction': 'given the code, document the function in a professional manner in the docs section', 
            'code': c.fn_code(fn),
            'docs': None,

        })
        # get the docs
        docs = model.generate(input)
        docs = self.process_response(docs)

        # add docs to the function
        c.add_docs(fn, docs)

        return docs
    
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

