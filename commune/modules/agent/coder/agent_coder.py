import commune as c
import json

class AgentCoder(c.Module):
    def document_fn(self,
             fn='agent.coder/call', 
             model = 'model.openai',
             **model_params
             ):
        '''
        ### Function Documentation
        
        #### Function Name: 
        call
        
        #### Parameters:
        - `fn` (str): The designated function name. Default value is 'agent.coder/call'.
        - `model` (str): The model identifier. Default value is 'model.openai'.
        
        #### Description:
        This function is designed to generate documentation for a given function by connecting to a model and sending a JSON-encoded request containing the function's code and instruction for documentation. The function handles the connection to the model, generates
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
    
    
    
    def document_module(self,
             module='agent.coder', 
             fns = ['document_fn'],
             model = 'model.openai',
             **model_params
             ):

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
        assert isinstance(response, str), f'Invalid docs type: {type(docs)}'
        
        return response

