
from typing import *
from copy import deepcopy
import inspect
from munch import Munch

class Schema:

    @classmethod
    def schema(cls,
                search = None,
                module = None,
                whitelist = None,
                fn = None,
                docs: bool = True,
                include_parents:bool = False,
                defaults:bool = True, cache=False) -> 'Schema':
        
        if '/' in str(search):
            module, fn = search.split('/')
            cls = cls.module(module)
        if isinstance(module, str):
            if '/' in module:
                module , fn = module.split('/')
            module = cls.module(module)

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
        if whitelist != None :
            schema = {k:v for k,v in schema.items() if k in whitelist}
        return schema
        

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
    def get_parents(cls, obj = None,recursive=True, avoid_classes=['object']) -> List[str]:
        obj = cls.resolve_object(obj)
        parents =  list(obj.__bases__)
        if recursive:
            for parent in parents:
                parent_parents = cls.get_parents(parent, recursive=recursive)
                if len(parent_parents) > 0:
                    for pp in parent_parents: 
                        if pp.__name__ not in avoid_classes:
                        
                            parents += [pp]
        return parents

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
                    x[k] = cls.munch2dict(v)

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
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True, 
                            version=2)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_schema = {}
        print(fn,'FAM')
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

   


    @classmethod
    def fn_info(cls, fn:str='test_fn') -> dict:
        r = {}
        code = cls.fn_code(fn)
        lines = code.split('\n')
        mode = 'self'
        if '@classmethod' in lines[0]:
            mode = 'class'
        elif '@staticmethod' in lines[0]:
            mode = 'static'
    
        start_line_text = None
        lines_before_fn_def = 0
        for l in lines:
            
            if f'def {fn}('.replace(' ', '') in l.replace(' ', ''):
                start_line_text = l
                break
            else:
                lines_before_fn_def += 1
            
        assert start_line_text != None, f'Could not find function {fn} in {cls.pypath()}'
        module_code = cls.code()
        start_line = cls.find_code_line(start_line_text, code=module_code) - lines_before_fn_def - 1
        end_line = start_line + len(lines)   # find the endline
        has_docs = bool('"""' in code or "'''" in code)
        filepath = cls.filepath()

        # start code line
        for i, line in enumerate(lines):
            
            is_end = bool(')' in line and ':' in line)
            if is_end:
                start_code_line = i
                break 

        
        return {
            'start_line': start_line,
            'end_line': end_line,
            'has_docs': has_docs,
            'code': code,
            'n_lines': len(lines),
            'hash': c.hash(code),
            'path': filepath,
            'start_code_line': start_code_line + start_line ,
            'mode': mode
            
        }
    


    @classmethod
    def find_code_line(cls, search:str, code:str = None):
        if code == None:
            code = cls.code() # get the code
        found_lines = [] # list of found lines
        for i, line in enumerate(code.split('\n')):
            if search in line:
                found_lines.append({'idx': i+1, 'text': line})
        if len(found_lines) == 0:
            return None
        elif len(found_lines) == 1:
            return found_lines[0]['idx']
        return found_lines
    


    @classmethod
    def attributes(cls):
        return list(cls.__dict__.keys())


    @classmethod
    def get_attributes(cls, search = None, obj=None):
        if obj is None:
            obj = cls
        if isinstance(obj, str):
            obj = c.module(obj)
        # assert hasattr(obj, '__dict__'), f'{obj} has no __dict__'
        attrs =  dir(obj)
        if search is not None:
            attrs = [a for a in attrs if search in a and callable(a)]
        return attrs
    

    
    def add_fn(self, fn, name=None):
        if name == None:
            name = fn.__name__
        assert not hasattr(self, name), f'{name} already exists'

        setattr(self, name, fn)

        return {
            'success':True ,
            'message':f'Added {name} to {self.__class__.__name__}'
        }
    

    add_attribute = add_attr = add_function = add_fn

    def metadata(self):
        schema = self.schema()
        return {fn: schema[fn] for fn in self.whitelist if fn not in self.blacklist and fn in schema}

    @classmethod
    def init_schema(cls):
        return cls.fn_schema('__init__')
    


    @classmethod
    def init_kwargs(cls):
        kwargs =  cls.fn_defaults('__init__')
        kwargs.pop('self', None)
        if 'config' in kwargs:
            if kwargs['config'] != None:
                kwargs.update(kwargs.pop('config'))
            del kwargs['config']
        if 'kwargs' in kwargs:
            if kwargs['kwargs'] != None:
                kwargs = kwargs.pop('kwargs')
            del kwargs['kwargs']

        return kwargs
    

    @classmethod
    def code(cls, module = None, search=None, *args, **kwargs):
        if '/' in str(module) or module in cls.fns():
            return cls.fn_code(module)
            
        module = cls.resolve_object(module)
        text =  cls.get_text( module.pypath(), *args, **kwargs)
        if search != None:
            find_lines = cls.find_lines(text=text, search=search)
            return find_lines
        return text
        


    pycode = code

    @classmethod
    def chash(cls,  *args, **kwargs):
        """
        The hash of the code, where the code is the code of the class (cls)
        """
        code = cls.code(*args, **kwargs)
        return c.hash(code)
    
    @classmethod
    def find_code_line(cls, search:str, code:str = None):
        if code == None:
            code = cls.code() # get the code
        found_lines = [] # list of found lines
        for i, line in enumerate(code.split('\n')):
            if search in line:
                found_lines.append({'idx': i+1, 'text': line})
        if len(found_lines) == 0:
            return None
        elif len(found_lines) == 1:
            return found_lines[0]['idx']
        return found_lines
    
    @classmethod
    def fn_info(cls, fn:str='test_fn') -> dict:
        r = {}
        code = cls.fn_code(fn)
        lines = code.split('\n')
        mode = 'self'
        if '@classmethod' in lines[0]:
            mode = 'class'
        elif '@staticmethod' in lines[0]:
            mode = 'static'
    
        start_line_text = None
        lines_before_fn_def = 0
        for l in lines:
            
            if f'def {fn}('.replace(' ', '') in l.replace(' ', ''):
                start_line_text = l
                break
            else:
                lines_before_fn_def += 1
            
        assert start_line_text != None, f'Could not find function {fn} in {cls.pypath()}'
        module_code = cls.code()
        start_line = cls.find_code_line(start_line_text, code=module_code) - lines_before_fn_def - 1
        end_line = start_line + len(lines)   # find the endline
        has_docs = bool('"""' in code or "'''" in code)
        filepath = cls.filepath()

        # start code line
        for i, line in enumerate(lines):
            
            is_end = bool(')' in line and ':' in line)
            if is_end:
                start_code_line = i
                break 

        
        return {
            'start_line': start_line,
            'end_line': end_line,
            'has_docs': has_docs,
            'code': code,
            'n_lines': len(lines),
            'hash': c.hash(code),
            'path': filepath,
            'start_code_line': start_code_line + start_line ,
            'mode': mode
            
        }
    

    @classmethod
    def set_line(cls, idx:int, text:str):
        code = cls.code()
        lines = code.split('\n')
        if '\n' in text:
            front_lines = lines[:idx]
            back_lines = lines[idx:]
            new_lines = text.split('\n')
            c.print(new_lines)
            lines = front_lines + new_lines + back_lines
        else:
            lines[idx-1] = text
        new_code = '\n'.join(lines)
        cls.put_text(cls.filepath(), new_code)
        return {'success': True, 'msg': f'Set line {idx} to {text}'}

    @classmethod
    def add_line(cls, idx=0, text:str = '',  module=None  ):
        """
        add line to an index of the module code
        """

        code = cls.code() if module == None else c.module(module).code()
        lines = code.split('\n')
        new_lines = text.split('\n') if '\n' in text else [text]
        lines = lines[:idx] + new_lines + lines[idx:]
        new_code = '\n'.join(lines)
        cls.put_text(cls.filepath(), new_code)
        return {'success': True, 'msg': f'Added line {idx} to {text}'}

    @classmethod
    def get_line(cls, idx):
        code = cls.code()
        lines = code.split('\n')
        assert idx < len(lines), f'idx {idx} is out of range for {len(lines)}'  
        line =  lines[max(idx, 0)]
        print(len(line))
        return line

    def hasfn(self, fn:str):
        return hasattr(self, fn) and callable(getattr(self, fn))
    




    @classmethod
    def has_fn(cls,fn_name, obj = None):
        if obj == None:
            obj = cls
        return callable(getattr(obj, fn_name, None))


    
    @classmethod
    def fn_defaults(cls, fn):
        """
        Gets the function defaults
        """
        fn = cls.get_fn(fn)
        function_defaults = dict(inspect.signature(fn)._parameters)
        for k,v in function_defaults.items():
            if v._default != inspect._empty and  v._default != None:
                function_defaults[k] = v._default
            else:
                function_defaults[k] = None

        return function_defaults
 
    @staticmethod
    def is_class(obj):
        '''
        is the object a class
        '''
        return type(obj).__name__ == 'type'


    @staticmethod
    def resolve_class(obj):
        '''
        resolve class of object or return class if it is a class
        '''
        if c.is_class(obj):
            return obj
        else:
            return obj.__class__
        


    @classmethod
    def has_var_keyword(cls, fn='__init__', fn_signature=None):
        if fn_signature == None:
            fn_signature = cls.resolve_fn(fn)
        for param_info in fn_signature.values():
            if param_info.kind._name_ == 'VAR_KEYWORD':
                return True
        return False
    

    
    @classmethod
    def fn_signature(cls, fn) -> dict: 
        '''
        get the signature of a function
        '''
        if isinstance(fn, str):
            fn = getattr(cls, fn)
        return dict(inspect.signature(fn)._parameters)
    
    get_function_signature = fn_signature
    @classmethod
    def is_arg_key_valid(cls, key='config', fn='__init__'):
        fn_signature = cls.fn_signature(fn)
        if key in fn_signature: 
            return True
        else:
            for param_info in fn_signature.values():
                if param_info.kind._name_ == 'VAR_KEYWORD':
                    return True
        
        return False
    

    
    @classmethod
    def self_functions(cls: Union[str, type], obj=None, search=None):
        '''
        Gets the self methods in a class
        '''
        obj = cls.resolve_object(obj)
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        if search != None:
            functions = [f for f in functions if search in f]
        return [k for k, v in signature_map.items() if 'self' in v]
    
    @classmethod
    def class_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = cls.resolve_object(obj)
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if 'cls' in v]
    
    class_methods = get_class_methods =  class_fns = class_functions

    @classmethod
    def static_functions(cls: Union[str, type], obj=None):
        '''
        Gets the self methods in a class
        '''
        obj = obj or cls
        functions =  cls.get_functions(obj)
        signature_map = {f:cls.get_function_args(getattr(obj, f)) for f in functions}
        return [k for k, v in signature_map.items() if not ('self' in v or 'cls' in v)]
    
    static_methods = static_fns =  static_functions




    @classmethod
    def property_fns(cls) -> bool:
        '''
        Get a list of property functions in a class
        '''
        return [fn for fn in dir(cls) if cls.is_property(fn)]
    

    # @classmethod
    # def get_parents(cls, obj=None):
    #     '''
    #     Get the parent classes of a class
    #     '''
    #     obj = cls.resolve_object(obj)
    #     return obj.__bases__
    
    parents = get_parents
    
    @classmethod
    def parent2functions(cls, obj=None):
        '''
        Get the parent classes of a class
        '''
        obj = cls.resolve_object(obj)
        parent_functions = {}
        for parent in cls.parents(obj):
            parent_functions[parent.__name__] = cls.get_functions(parent)
        return parent_functions
    


    @classmethod
    def get_functions(cls, obj: Any = None,
                      search = None,
                      include_parents:bool=False, 
                      include_hidden:bool = False) -> List[str]:
        '''
        Get a list of functions in a class
        
        Args;
            obj: the class to get the functions from
            include_parents: whether to include the parent functions
            include_hidden:  whether to include hidden functions (starts and begins with "__")
        '''
        
        obj = cls.resolve_object(obj)
        functions = []
        child_functions = list(obj.__dict__.keys())
        parent_functions = []


        if cls.is_root_module():
            include_parents = True
            
        if include_parents:
            parent_functions = cls.get_parent_functions(obj)

 

        for fn_name in (child_functions + parent_functions):
            if search != None and search not in fn_name :
                continue
            
            # skip hidden functions if include_hidden is False
            if not include_hidden:
                if ((fn_name.startswith('__') or fn_name.endswith('_'))):
                    if fn_name != '__init__':
                        continue

            # if the function is in the parent class, skip it
            if not include_parents:
                if fn_name in parent_functions:
                    continue



            fn_obj = getattr(obj, fn_name)

            # if the function is a property, skip it
            if cls.is_property(fn_obj):
                continue
            # if the function is callable, include it
            if callable(fn_obj):
                functions.append(fn_name)

        functions = list(set(functions))     
            
        return functions

    

    @classmethod
    def is_property(cls, fn: 'Callable') -> bool:
        '''
        is the function a property
        '''
        try:
            fn = cls.get_fn(fn, ignore_module_pattern=True)
        except :
            return False

        return isinstance(fn, property)

    def is_fn_self(self, fn):
        fn = self.resolve_fn(fn)
        return hasattr(fn, '__self__') and fn.__self__ == self



    @classmethod
    def get_fn(cls, fn:str, init_kwargs = None):
        """
        Gets the function from a string or if its an attribute 
        """
        if isinstance(fn, str):
            if ':' in fn or '/' in fn:
                module, fn = fn.split(':')
                cls = cls.get_module(module)
            try:
                fn =  getattr(cls, fn)
            except:
                init_kwargs = init_kwargs or {}
                fn = getattr(cls(**init_kwargs), fn)

        if callable(fn) or isinstance(fn, property):
            pass

        return fn
    

        
    @classmethod
    def self_functions(cls, search = None):
        fns =  cls.classify_fns(cls)['self']
        if search != None:
            fns = [f for f in fns if search in f]
        return fns
    
    

    @classmethod
    def classify_fns(cls, obj= None, mode=None):
        method_type_map = {}
        obj = obj or c.module(obj)
        if isinstance(obj, str):
            obj = c.module(obj)
        for attr_name in dir(obj):
            method_type = None
            try:
                method_type = cls.classify_fn(getattr(obj, attr_name))
            except Exception as e:
                continue
        
            if method_type not in method_type_map:
                method_type_map[method_type] = []
            method_type_map[method_type].append(attr_name)
        if mode != None:
            method_type_map = method_type_map[mode]
        return method_type_map


    @classmethod
    def get_function_args(cls, fn) -> List[str]:
        """
        get the arguments of a function
        params:
            fn: the function
        
        """
        if not callable(fn):
            fn = cls.get_fn(fn)

        try:
            args = inspect.getfullargspec(fn).args
        except Exception as e:
            args = []
        return args

    
    @classmethod
    def has_function_arg(cls, fn, arg:str):
        args = cls.get_function_args(fn)
        return arg in args

    
    fn_args = get_fn_args =  get_function_args
    
    @classmethod
    def classify_fn(cls, fn):
        
        if not callable(fn):
            fn = cls.get_fn(fn)
        if not callable(fn):
            return None
        args = cls.get_function_args(fn)
        if len(args) == 0:
            return 'static'
        elif args[0] == 'self':
            return 'self'
        else:
            return 'class'
        
    

    @classmethod
    def python2types(cls, d:dict)-> dict:
        return {k:str(type(v)).split("'")[1] for k,v in d.items()}
    



    @classmethod
    def fn2str(cls,search = None,  code = True, defaults = True, **kwargs):
        fns = cls.fns(search=search)
        fn2str = {}
        for fn in fns:
            fn2str[fn] = cls.fn_code(fn)
            
        return fn2str
    @classmethod
    def fn2hash(cls, fn=None , mode='sha256', **kwargs):
        fn2hash = {}
        for k,v in cls.fn2str(**kwargs).items():
            fn2hash[k] = c.hash(v,mode=mode)
        if fn:
            return fn2hash[fn]
        return fn2hash
    
    @classmethod
    def module2fn2str(self, code = True, defaults = False, **kwargs):
        module2fn2str = {  }
        for module in c.modules():
            try:
                module_class = c.module(module)
                if hasattr(module_class, 'fn2str'):
                    module2fn2str[module] = module_class.fn2str(code = code,                                          defaults = defaults, **kwargs)
            except:
                pass
        return module2fn2str

    # TAG CITY     
 

    # TAG CITY     
    @classmethod
    def get_parent_functions(cls, obj = None, include_root = True):
        import inspect
        functions = []
        obj = obj or cls
        
        parents = cls.get_parents(obj)
        for parent in parents:
            for name, member in parent.__dict__.items():
                if not name.startswith('__'):
                    functions.append(name)
        if cls.is_root_module():
            include_root = True
        if not include_root:
            root_fns = cls.root_fns()
            functions = [f for f in functions if f not in root_fns]
        return functions
    parent_functions = get_parent_functions

    @classmethod
    def get_child_functions(cls, obj=None):
        obj = cls.resolve_object(obj)
        
        methods = []
        for name, member in obj.__dict__.items():
            if inspect.isfunction(member) and not name.startswith('__'):
                methods.append(name)
        
        return methods
    
    child_functions = get_child_functions
    

    @classmethod
    def locals2kwargs(cls,locals_dict:dict, kwargs_keys=['kwargs']) -> dict:
        locals_dict = locals_dict or {}
        kwargs = locals_dict or {}
        kwargs.pop('cls', None)
        kwargs.pop('self', None)

        assert isinstance(kwargs, dict), f'kwargs must be a dict, got {type(kwargs)}'
        
        # These lines are needed to remove the self and cls from the locals_dict
        for k in kwargs_keys:
            kwargs.update( locals_dict.pop(k, {}) or {})

        return kwargs
    
    
    get_kwargs = get_params = locals2kwargs 



    @classmethod
    def transfer_fn_code(cls, module1= 'module',
                        fn_prefix = 'ray_',
                        module2 = 'ray',
                        refresh = False):

        module1 = c.module(module1)
        module2 = c.module(module2)
        module1_fn_code_map = module1.fn2code(fn_prefix)
        module2_code = module2.code()
        module2_fns = module2.fns()
        filepath = module2.filepath()
        for fn_name, fn_code in module1_fn_code_map.items():
            print(f'adding {fn_name}')
            print('fn_code', fn_code)
            if fn_name in module2_fns:
                if refresh:
                    module2_code = module2_code.replace(module2_fns[fn_name], '')
                else:
                    print(f'fn_name {fn_name} already in module2_fns {module2_fns}')

            module2_code += '\n'
            module2_code += '\n'.join([ '    ' + line for line in fn_code.split('\n')])
            module2_code += '\n'
        cls.put_text(filepath, module2_code)

        return {'success': True, 'module2_code': module2_code, 'module2_fns': module2_fns, 'module1_fn_code_map': module1_fn_code_map}


    @classmethod
    def find_classes(cls, path):
        code = cls.get_text(path)
        classes = []
        for line in code.split('\n'):
            if all([s in line for s in ['class ', '(', '):']]):
                classes.append(line.split('class ')[-1].split('(')[0].strip())
        return [c for c in classes]
    



    def info(self , 
             module = None,
             features = ['schema', 'namespace', 'commit_hash', 'hardware','attributes','functions'], 
             lite_features = ['name', 'address', 'schema', 'key', 'description'],
             lite = True,
             cost = False,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        if lite:
            features = lite_features
            
        if module != None:
            if isinstance(module, str):
                module = self.module(module)()
            self = module  
            
        info = {}

        if 'schema' in features:
            info['schema'] = self.schema(defaults=True, include_parents=True)
            info['schema'] = {k: v for k,v in info['schema'].items() if k in self.whitelist}
        if 'namespace' in features:
            info['namespace'] = self.namespace(network='local')
        if 'hardware' in features:
            info['hardware'] = self.hardware()
        if 'attributes' in features:
            info['attributes'] = self.attributes()
        if 'functions' in features:
            info['functions']  =  self.whitelist
        if 'name' in features:
            info['name'] = self.server_name() if callable(self.server_name) else self.server_name # get the name of the module
        if 'path' in features:
            info['path'] = self.module_name() # get the path of the module
        if 'address' in features:
            info['address'] = self.address
        if 'key' in features:    
            info['key'] = self.key.ss58_address
        if 'code_hash' in features:
            info['code_hash'] = self.chash() # get the hash of the module (code)
        if 'commit_hash' in features:
            info['commit_hash'] = self.commit_hash()
        if 'description' in features:
            info['description'] = self.description

        self.put_json('info', info)
        if cost:
            if hasattr(self, 'cost'):
                info['cost'] = self.cost
        return info
        
    help = info
