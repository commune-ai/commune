import commune as c
import json
import os
import glob 
from typing import *
from copy import deepcopy
import inspect
from munch import Munch

class Code(c.Module):

    def file2text(self, path = './', relative=False,  **kwargs):
        path = os.path.abspath(path)
        file2text = {}
        for file in glob.glob(path, recursive=True):
            with open(file, 'r') as f:
                content = f.read()
                file2text[file] = content
        return file2text
                



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

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                with open(file_path, 'r') as f:
                    code = f.read()
                    code_dict[relative_path] = code

        return code_dict
    

        
    @classmethod
    def process_kwargs(cls, kwargs:dict, fn_schema:dict):
        
        for k,v in kwargs.items():
            if v == 'None':
                v = None
            
            if isinstance(v, str):
                if v.startswith('[') and v.endswith(']'):
                    if len(v) > 2:
                        v = eval(v)
                    else:
                        v = []

                elif v.startswith('{') and v.endswith('}'):

                    if len(v) > 2:
                        v = c.jload(v)
                    else:
                        v = {}               
                elif k in fn_schema['input'] and fn_schema['input'][k] == 'str':
                    if v.startswith("f'") or v.startswith('f"'):
                        v = c.ljson(v)
                    else:
                        v = v

                elif fn_schema['input'][k] == 'float':
                    v = float(v)

                elif fn_schema['input'][k] == 'int':
                    v = int(v)

                elif k == 'kwargs':
                    continue
                elif v == 'NA':
                    assert k != 'NA', f'Key {k} not in default'
                elif v in ['True', 'False']:
                    v = eval(v)
                elif c.is_int(v):
                    v = eval(v)
                else:
                    v = v
            
            kwargs[k] = v

        return kwargs
    

    
    @classmethod
    def python2str(cls, input):
        input = deepcopy(input)
        input_type = type(input)
        if input_type == str:
            return input
        if input_type in [dict]:
            input = json.dumps(input)
        elif input_type in [bytes]:
            input = cls.bytes2str(input)
        elif input_type in [list, tuple, set]:
            input = json.dumps(list(input))
        elif input_type in [int, float, bool]:
            input = str(input)
        return input
    

    
    # JSON2BYTES
    @classmethod
    def dict2str(cls, data: str) -> str:
        return json.dumps(data)
    
    @classmethod
    def dict2bytes(cls, data: str) -> bytes:
        return cls.str2bytes(cls.json2str(data))
    
    @classmethod
    def bytes2dict(cls, data: bytes) -> str:
        data = cls.bytes2str(data)
        return json.loads(data)
    

    # BYTES LAND
    
    # STRING2BYTES
    @classmethod
    def str2bytes(cls, data: str, mode: str = 'hex') -> bytes:
        if mode in ['utf-8']:
            return bytes(data, mode)
        elif mode in ['hex']:
            return bytes.fromhex(data)
    
    @classmethod
    def bytes2str(cls, data: bytes, mode: str = 'utf-8') -> str:
        
        if hasattr(data, 'hex'):
            return data.hex()
        else:
            if isinstance(data, str):
                return data
            return bytes.decode(data, mode)


    @classmethod
    def str2python(cls, input)-> dict:
        assert isinstance(input, str), 'input must be a string, got {}'.format(input)
        try:
            output_dict = json.loads(input)
        except json.JSONDecodeError as e:
            return input

        return output_dict
    

    
    @classmethod
    def fn2code(cls, search=None, module=None)-> Dict[str, str]:
        module = module if module else cls
        functions = module.fns(search)
        fn_code_map = {}
        for fn in functions:
            c.print(f'fn: {fn}')
            try:
                fn_code_map[fn] = module.fn_code(fn)
            except Exception as e:
                c.print(f'Error: {e}', color='red')
        return fn_code_map
    

    
    @classmethod
    def fn_code(cls,fn:str, 
                detail:bool=False, 
                seperator: str = '/'
                ) -> str:
        '''
        Returns the code of a function
        '''
        try:
            if isinstance(fn, str):
                if seperator in fn:
                    module_path, fn = fn.split(seperator)
                    module = c.module(module_path)
                    fn = getattr(module, fn)
                else:
                    fn = getattr(cls, fn)
            
            
            code_text = inspect.getsource(fn)
            text_lines = code_text.split('\n')
            if 'classmethod' in text_lines[0] or 'staticmethod' in text_lines[0] or '@' in text_lines[0]:
                text_lines.pop(0)

            assert 'def' in text_lines[0], 'Function not found in code'
            start_line = cls.find_code_line(search=text_lines[0])
            fn_code = '\n'.join([l[len('    '):] for l in code_text.split('\n')])
            if detail:
                fn_code =  {
                    'text': fn_code,
                    'start_line': start_line ,
                    'end_line':  start_line + len(text_lines)
                }
        except Exception as e:
            c.print(f'Error: {e}', color='red')
            fn_code = None
                    
        return fn_code
    

    @classmethod
    def is_generator(cls, obj):
        """
        Is this shiz a generator dawg?
        """
        if isinstance(obj, str):
            if not hasattr(cls, obj):
                return False
            obj = getattr(cls, obj)
        if not callable(obj):
            result = inspect.isgenerator(obj)
        else:
            result =  inspect.isgeneratorfunction(obj)
        return result



    @staticmethod
    def get_parents(obj) -> List[str]:
        cls = c.resolve_class(obj)
        return list(cls.__mro__[1:-1])

    @staticmethod
    def get_parent_functions(cls) -> List[str]:
        parent_classes = c.get_parents(cls)
        function_list = []
        for parent in parent_classes:
            function_list += c.get_functions(parent)

        return list(set(function_list))
    

    
    def repo2module(self, repo:str, name=None, template_module='demo', **kwargs):
        if not repo_path.startswith('/') and not repo_path.startswith('.') and not repo_path.startswith('~'):
            repo_path = os.path.abspath('~/' + repo_path)
        assert os.path.isdir(repo_path), f'{repo_path} is not a directory, please clone it'
        c.add_tree(repo_path)
        template_module = c.module(template_module)
        code = template_module.code()

        # replace the template module with the name
        name = name or repo_path.split('/')[-1] 
        assert not c.module_exists(name), f'{name} already exists'
        code_lines = code.split('\n')
        for i, line in enumerate(code_lines):
            if 'class' in line and 'c.Module' in line:
                class_name = line.split('class ')[-1].split('(')[0]
                code_lines[i] = line.replace(class_name, name)
                break
        code = '\n'.join(code_lines)

        module_path = repo_path + '/module.py'

        # write the module code
        c.put_text(code, module_path)

        # build the tree
        c.build_tree(update=True)




    @classmethod
    def timefn(cls, fn, *args, **kwargs):
        fn = cls.get_fn(fn)
        if isinstance(fn, str):
            if '/' in fn:
                module, fn = fn.split('/')
                module = c.module(module)
            else:
                module = cls
            if module.classify_fn(fn) == 'self':
                module = cls()
            fn = getattr(module, fn)
        
        t1 = c.time()
        result = fn(*args, **kwargs)
        t2 = c.time()

        return {'time': t2 - t1}
    


    
    @classmethod
    def find_python_classes(cls, path:str , class_index:int=0, search:str = None, start_lines:int=2000):
        import re
        path = cls.resolve_path(path)
        if os.path.isdir(path):
            file2classes = {}
            for f in c.glob(path):
                if f.endswith('.py'):
                    try:
                        file2classes[f] = cls.find_python_classes(f, class_index=class_index, search=search, start_lines=start_lines)
                    except Exception as e:
                        c.print(f'Error: {e}', color='red')
            return file2classes
        # read the contents of the Python script file
        python_script = cls.readlines(path, end_line = start_lines, resolve=False)
        class_names  = []
        lines = python_script.split('\n')

        # c.print(python_script)
        
        for line in lines:
            key_elements = ['class ', '(', '):']
            has_class_bool = all([key_element in line for key_element in key_elements])

            if has_class_bool:
                if  search != None:
                    if isinstance(search, str):
                        search = [search]
                    if not any([s in line for s in search]):
                        continue
                        
                class_name = line.split('class ')[-1].split('(')[0].strip()
                class_names.append(class_name)
                
        # return the class names
        return class_names
    

    @classmethod
    def find_functions(cls, path):
        code = c.get_text(path)
        functions = []
        for line in code.split('\n'):
            if line.startswith('def '):
                if all([s in line for s in ['def ', '(', '):']]):
                    functions.append(line.split('def ')[-1].split('(')[0].strip())
        return functions


    def file2classes(self, path:str = None, search:str = None, start_lines:int=2000):
        return self.find_python_classes(path=path, search=search, start_lines=start_lines)



    @classmethod
    def get_class_name(cls, obj = None) -> str:
        obj = obj if obj != None else cls
        if not cls.is_class(obj):
            obj = type(obj)
        return obj.__name__

    @staticmethod
    def try_n_times(fn, max_trials:int=10, args:list=[],kwargs:dict={}):
        assert isinstance(fn, callable)
        for t in range(max_trials):
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                continue
        raise(e)



    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> Munch:
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = c.dict2munch(v)
            x = Munch(x)
        return x 

    @classmethod
    def munch2dict(cls, x:Munch, recursive:bool=True)-> dict:
        '''
        Turn munch object  into dictionary
        '''
        if isinstance(x, Munch):
            x = dict(x)
            for k,v in x.items():
                if isinstance(v, Munch) and recursive:
                    x[k] = c.munch2dict(v)

        return x 

    
    
    
    @classmethod
    def munch(cls, x:Dict) -> Munch:
        '''
        Converts a dict to a munch
        '''
        return cls.dict2munch(x)
    


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
