
from typing import *
from copy import deepcopy
import inspect
from munch import Munch

class Schema:

    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.get_fn(fn)
        if not hasattr(fn, '__annotations__'):
            return {}
        return fn.__annotations__


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
                

    @classmethod
    def resolve_object(cls, obj) -> Any:
        return obj or cls
    


    @classmethod
    def get_function_default_map(cls, obj:Any= None, include_parents=False) -> Dict[str, Dict[str, Any]]:
        obj = cls.resolve_object(obj)
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
    


    
    @classmethod
    def fn2code(cls, search=None, module=None)-> Dict[str, str]:
        module = module if module else cls
        functions = module.fns(search)
        fn_code_map = {}
        for fn in functions:
            try:
                fn_code_map[fn] = module.fn_code(fn)
            except Exception as e:
                print(f'Error: {e}')
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
            fn = cls.get_fn(fn)
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
            print(f'Error: {e}')
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

    @classmethod
    def get_parents(cls, obj) -> List[str]:
        cls = cls.resolve_object(obj)
        return list(cls.__mro__[1:-1])

    @staticmethod
    def get_parent_functions(cls) -> List[str]:
        parent_classes = cls.get_parents(cls)
        function_list = []
        for parent in parent_classes:
            function_list += cls.get_functions(parent)
        return list(set(function_list))

    @classmethod
    def get_class_name(cls, obj = None) -> str:
        obj = cls or obj
        if not cls.is_class(obj):
            obj = type(obj)
        return obj.__name__
    
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
 
        fn_args = cls.get_function_args(fn)
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
    def fn_signature_map(cls, obj=None, include_parents:bool = False):
        function_signature_map = {}
        obj = cls.resolve_object(obj)
        for f in cls.get_functions(obj = obj, include_parents=include_parents):
            if f.startswith('__') and f.endswith('__'):
                if f in ['__init__']:
                    pass
                else:
                    continue
            if not hasattr(cls, f):
                continue
            if callable(getattr(cls, f )):
                function_signature_map[f] = {k:str(v) for k,v in cls.get_function_signature(getattr(cls, f )).items()}        
        
    
        return function_signature_map


    @classmethod
    def dict2munch(cls, x:dict, recursive:bool=True)-> Munch:
        '''
        Turn dictionary into Munch
        '''
        if isinstance(x, dict):
            for k,v in x.items():
                if isinstance(v, dict) and recursive:
                    x[k] = cls.dict2munch(v)
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
    


    @classmethod
    def get_function_annotations(cls, fn):
        fn = cls.get_fn(fn)
        if not hasattr(fn, '__annotations__'):
            return {}
        return fn.__annotations__



    @classmethod
    def get_fn(cls, fn:str):
        

        """
        
        Gets the function from a string or if its an attribute 
        """

        if isinstance(fn, str):
            fn = getattr(cls, fn)
        elif callable(fn):
            pass
        elif isinstance(fn, property):
            pass
        else:
            raise ValueError(f'fn must be a string or callable, got {type(fn)}')
        # assert callable(fn), 'Is not callable'
        return fn
    


    






    @classmethod
    def schema(cls,
                search = None,
                module = None,
                fn = None,
                docs: bool = True,
                include_parents:bool = False,
                defaults:bool = True, cache=False) -> 'Schema':

        if '/' in str(search):
            module, fn = search.split('/')
            cls = c.module(module)
        if isinstance(module, str):
            if '/' in module:
                module , fn = module.split('/')
            module = c.module(module)

        module = module or cls
        schema = {}
        fns = module.get_functions(include_parents=include_parents)
        for fn in fns:
            if search != None and search not in fn:
                continue
            if callable(getattr(module, fn )):
                schema[fn] = cls.fn_schema(fn, defaults=defaults,docs=docs)        

        # sort by keys
        schema = dict(sorted(schema.items()))

        return schema
        


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
 
        fn_args = cls.get_function_args(fn)
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
    

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__

   