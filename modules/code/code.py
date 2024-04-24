import commune as c
import json

class Coder(c.Module):
    def comment(self,
             fn='coder/call', 
             model = 'model.openai',
             timeout=40,
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
        model = c.m('model.openrouter')()
        input = json.dumps({
            'instruction': 'given the code, document the function in a professional manner in the docs section', 
            'code': c.fn_code(fn),
            'docs': None,
        })
        # get the docs
        docs = model.generate(input, timeout=timeout)
        docs = self.process_response(docs)

        # add docs to the function
        self.add_docs(fn, docs)

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


    @classmethod
    def add_fn_code(cls, fn:str='test_fn', code:str = None):
        fn_info = cls.fn_info(fn)
        start_line = fn_info["start_line"]
        end_line = fn_info["end_line"]
        module_code = cls.code()
        lines = module_code.split('\n')
        if code == None:
            code = ''
        new_lines = lines[:start_line] + [code] + lines[end_line:]
        new_code = '\n'.join(new_lines)
        return c.put_text(cls.filepath(), new_code)

    @classmethod
    def rm_docs(cls, fn:str='rm_docs'):
        """
        sup
        """

        doc_info = cls.fn_docs(fn, include_quotes=True, return_dict=True)
        
        doc_idx_bounds = doc_info['idx_bounds']

        if doc_idx_bounds == None:
            return None

        fn_info = cls.fn_info(fn)

        fn_code = fn_info['code']
        
        before_comment_code = fn_code.split('\n')[:doc_idx_bounds[0] - 2]
       
        after_comment_code = fn_code.split('\n')[doc_idx_bounds[1]:]
        
        new_fn_code = '\n'.join(before_comment_code + after_comment_code)
        
        return c.add_fn_code(fn=fn, code=new_fn_code)
    
    def rm_fn(self, fn:str='rm_fn'):
        return self.add_fn_code(fn, code='')
    

    @classmethod 
    def fn_docs(cls, fn:str='test_fn2', include_quotes=False, return_dict=False):
        '''
        This is a document
        '''
        if '/' in fn:
            cls = c.module(fn.split('/')[0])
            fn = fn.split('/')[1]
    
        fn_info = cls.fn_info(fn)
        start_line = fn_info["start_code_line"]
        '''
        sup
        '''
        end_line = fn_info["end_line"]
        code = cls.code()
        lines = code.split('\n')
        comment_idx_bounds = []

        for i, line in enumerate(lines[start_line:end_line]):

            comment_bounds = ['"""', "'''"]
            for comment_bound in comment_bounds:
                if  comment_bound in line:
                    comment_idx_bounds.append(i)
            if len(comment_idx_bounds) == 2:
                break

        if len(comment_idx_bounds) == 0:
            return {
                'idx_bounds': None,
                'text': None,
            }
        

        start_line_shift = -1 if include_quotes else 0
        end_line_shift = 1 if include_quotes else 0
        idx_bounds = [start_line+comment_idx_bounds[0] + start_line_shift, start_line+comment_idx_bounds[1]+ end_line_shift + 1]
        comment_text = '\n'.join(lines[idx_bounds[0]:idx_bounds[1]])

        if return_dict:
            return {
            'idx_bounds': comment_idx_bounds,
            'text': comment_text,
            }
        return comment_text

    @classmethod
    def add_lines(cls, idx=0, n=1 ):
        for i in range(n):
            cls.add_line(idx=idx)

    @classmethod
    def add_docs(cls, fn='add_docs', comment="This is a document"):
        '''
        This is a document
        '''
        '''
        This is a document
        '''
        if '/' in fn:
            cls = c.module(fn.split('/')[0])
            fn = fn.split('/')[1]

        
        fn_info = cls.fn_info(fn)
        start_line = fn_info["start_code_line"] + 1
        tab_space = "        "
        cls.add_line(idx=start_line, text=tab_space+"'''")
        c.print(comment)
        for i, line in enumerate(comment.split('\n')):
            cls.add_line(idx=start_line+i+1, text=tab_space + line)
        cls.add_line(idx=start_line+len(comment.split('\n')) + 1, text=tab_space + "'''")
        
    @classmethod
    def is_empty_line(cls, idx):
        line = cls.get_line(idx)
        return len(line.strip()) == 0

    @classmethod
    def get_code_line(cls, module=None, idx:int = 0, code:str = None ):
        cls = c.module(module)
        if code == None:

            code = cls.code() # get the code
        lines = code.split('\n')
        assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'
        return lines[idx]
    
    @classmethod
    def imported_modules(cls, module=None):
        # get the text
        text = c.module(module or cls.path()).code()
        imported_modules = []
        module2line = {}
        for i,line in enumerate(text.split('\n')):
            if 'c.module(' in line:
                imported_module= line.split('c.module(')[1].split(')')[0]
                imported_module = imported_module.replace("'",'').replace('"','')
                module2line[imported_module] = i

        # sort the modules by line number
        modules = c.modules()
        module2line = {k: v for k, v in sorted(module2line.items(), key=lambda item: item[1]) if k in modules}

        return module2line