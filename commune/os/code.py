import commune as c
import json
import os
import glob 
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

    def file2text(self, path = './', relative=False,  **kwargs):
        path = os.path.abspath(path)
        file2text = {}
        for file in glob.glob(path, recursive=True):
            with open(file, 'r') as f:
                content = f.read()
                file2text[file] = content
        return file2text
                

    def file2fns(self, filepath = '~/commune/utils/dict.py'):
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
    

    @staticmethod
    def get_files_code(directory):
        code_dict = {}
        import glob 
        directory = os.path.abspath(directory)

        for file in glob.glob(directory + '/**', recursive=True):

            relative_path = file.split(directory)[1]
            if os.path.isdir(file):
                continue
            try:
                with open(file, 'r') as f:
                    code = f.read()
                    code_dict[file] = code
            except Exception as e:
                print(e)
                

        return code_dict

    @classmethod
    def eval(cls, module, vali=None,  **kwargs):
        vali = c.module('vali')() if vali == None else c.module(vali)
        return c.eval(module, **kwargs)
    


        
    @classmethod
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True, 
                            version=2)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_schema = {}
        fn = cls.get_fn(fn)
        fn_schema['input']  = cls.get_function_annotations(fn=fn)
        
        for k,v in fn_schema['input'].items():
            v = str(v)
            if v.startswith('<class'):
                fn_schema['input'][k] = v.split("'")[1]
            elif v.startswith('typing.'):
                fn_schema['input'][k] = v.split(".")[1].lower()
            else:
                fn_schema['input'][k] = v
                
        fn_schema['output'] = fn_schema['input'].pop('return', {})
        
        if docs:         
            fn_schema['docs'] =  fn.__doc__ 
        if code:
            fn_schema['code'] = cls.fn_code(fn)
 
        fn_args = c.get_function_args(fn)
        fn_schema['type'] = 'static'
        for arg in fn_args:
            if arg not in fn_schema['input']:
                fn_schema['input'][arg] = 'NA'
            if arg in ['self', 'cls']:
                fn_schema['type'] = arg
                fn_schema['input'].pop(arg)
                if 'default' in fn_schema:
                    fn_schema['default'].pop(arg, None)


        if defaults:
            fn_schema['default'] = cls.fn_defaults(fn=fn) 
            for k,v in fn_schema['default'].items(): 
                if k not in fn_schema['input'] and v != None:
                    fn_schema['input'][k] = type(v).__name__ if v != None else None
           
        if version == 1:
            pass
        elif version == 2:
            defaults = fn_schema.pop('default', {})
            fn_schema['input'] = {k: {'type':v, 'default':defaults.get(k)} for k,v in fn_schema['input'].items()}
        else:
            raise Exception(f'Version {version} not implemented')
                

        return fn_schema
    



    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.get_fn(fn)
        if not hasattr(fn, '__annotations__'):
            return {}
        return fn.__annotations__
    
    @classmethod
    def lock_file(cls, f):
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        return f
    
    @classmethod
    def unlock_file(cls, f):
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)
        return f


    def server2fn(self, *args, **kwargs ):
        servers = c.servers(*args, **kwargs)
        futures = []
        server2fn = {}
        for s in servers:
            server2fn[s] = c.submit(f'{s}/schema', kwargs=dict(code=True))
        futures = list(server2fn.values())
        fns = c.wait(futures,timeout=10)
        for s, f in zip(servers, fns):
            server2fn[s] = f
        return server2fn
    

    
    @classmethod
    def remove_number_from_word(cls, word:str) -> str:
        while word[-1].isdigit():
            word = word[:-1]
        return word
    
    @classmethod
    def determine_type(cls, x):
        if x.lower() == 'null' or x == 'None':
            return None
        elif x.lower() in ['true', 'false']:
            return bool(x.lower() == 'true')
        elif x.startswith('[') and x.endswith(']'):
            # this is a list
            try:
                
                list_items = x[1:-1].split(',')
                # try to convert each item to its actual type
                x =  [cls.determine_type(item.strip()) for item in list_items]
                if len(x) == 1 and x[0] == '':
                    x = []
                return x
       
            except:
                # if conversion fails, return as string
                return x
        elif x.startswith('{') and x.endswith('}'):
            # this is a dictionary
            if len(x) == 2:
                return {}
            try:
                dict_items = x[1:-1].split(',')
                # try to convert each item to a key-value pair
                return {key.strip(): cls.determine_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
            except:
                # if conversion fails, return as string
                return x
        else:
            # try to convert to int or float, otherwise return as string
            try:
                return int(x)
            except ValueError:
                try:
                    return float(x)
                except ValueError:
                    return x
