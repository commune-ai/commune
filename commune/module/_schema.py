
from typing import *
from copy import deepcopy
import inspect
from munch import Munch

class Schema:
    whitelist = []

    _schema = None
    def schema(self,
                search = None,
                docs: bool = True,
                defaults:bool = True, 
                cache=True) -> 'Schema':
        schema = {}
        if cache and self._schema != None:
            return self._schema
        fns = self.public_functions()
        for fn in fns:
            if search != None and search not in fn:
                continue
            if callable(getattr(self, fn )):
                schema[fn] = self.fn_schema(fn, defaults=defaults,docs=docs)        
        # sort by keys
        schema = dict(sorted(schema.items()))
        if cache:
            self._schema = schema

        return schema
    @classmethod
    def get_schema(cls,
                module = None,
                search = None,
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
        fns = module.get_functions()
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
            fn_code = '\n'.join([l[len('    '):] for l in code_text.split('\n')])
            assert 'def' in text_lines[0], 'Function not found in code'

            if detail:
                start_line = cls.find_code_line(search=text_lines[0])
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
    def fn_schema(cls, fn:str,
                            defaults:bool=True,
                            code:bool = False,
                            docs:bool = True, **kwargs)->dict:
        '''
        Get function schema of function in cls
        '''
        fn_schema = {}
        fn = cls.get_fn(fn)
        input_schema  = cls.fn_signature(fn)
        for k,v in input_schema.items():
            v = str(v)
            if v.startswith('<class'):
                input_schema[k] = v.split("'")[1]
            elif v.startswith('typing.'):
                input_schema[k] = v.split(".")[1].lower()
            else:
                input_schema[k] = v

        fn_schema['input'] = input_schema
        fn_schema['output'] = input_schema.pop('return', {})

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

        if defaults:
            fn_defaults = cls.fn_defaults(fn=fn) 
            for k,v in fn_defaults.items(): 
                if k not in fn_schema['input'] and v != None:
                    fn_schema['input'][k] = type(v).__name__ if v != None else None

        fn_schema['input'] = {k: {'type':v, 'default':fn_defaults.get(k)} for k,v in fn_schema['input'].items()}

        return fn_schema



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
            'hash': cls.hash(code),
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
    def lines_of_code(cls, code:str=None):
        if code == None:
            code = cls.code()
        return len(code.split('\n'))

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
            parent_functions = cls.parent_functions(obj)

 

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
    def functions(cls, search = None, include_parents = False):
        return cls.get_functions(search=search, include_parents=include_parents)

    fns = functions
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
            if '/' in fn:
                module, fn = fn.split('/')
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

    # TAG CITY     
    @classmethod
    def parent_functions(cls, obj = None, include_root = True):
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

    @classmethod
    def child_functions(cls, obj=None):
        obj = cls.resolve_object(obj)
        
        methods = []
        for name, member in obj.__dict__.items():
            if inspect.isfunction(member) and not name.startswith('__'):
                methods.append(name)
        
        return methods

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



    def info(self , 
             module = None,
             lite_features = ['name', 'address', 'schema', 'key', 'description'],
             lite = True,
             cost = False,
             **kwargs
             ) -> Dict[str, Any]:
        '''
        hey, whadup hey how is it going
        '''
        info = self.metadata()
        info['name'] = self.server_name or self.module_name()
        info['address'] = self.address
        info['key'] = self.key.ss58_address
        return info
    


    @classmethod
    def endpoint(cls, 
                 cost=1, # cost per call 
                 user2rate : dict = None, 
                 rate_limit : int = 100, # calls per minute
                 timestale : int = 60,
                 cost_keys = ['cost', 'w', 'weight'],
                 **kwargs):
        
        for k in cost_keys:
            if k in kwargs:
                cost = kwargs[k]
                break

        def decorator_fn(fn):
            metadata = {
                **Schema.fn_schema(fn),
                'cost': cost,
                'rate_limit': rate_limit,
                'user2rate': user2rate,                
            }
            import commune as c
            c.print(f'Adding metadata to {fn.__name__} : {metadata}')
            fn.__dict__['__metadata__'] = metadata

            return fn

        return decorator_fn
    

    def is_endpoint(self, fn) -> bool:
        if isinstance(fn, str):
            fn = getattr(self, fn)
        return hasattr(fn, '__metadata__')

    
    def public_functions(self, search=None, include_helper_functions = True):
        endpoints = []  
        if include_helper_functions:
            endpoints += self.helper_functions

        for f in dir(self):
            try:
                if search != None:
                    if search not in f:
                        continue
                fn_obj = getattr(self, f) # you need to watchout for properties
                is_endpoint = hasattr(fn_obj, '__metadata__')
                if is_endpoint:
                    endpoints.append(f)
            except:
                print(f)
        if hasattr(self, 'whitelist'):
            endpoints += self.whitelist
            endpoints = list(set(endpoints))

        return endpoints

    get_whitelist = endpoints = public_functions
    

    def cost_fn(self, fn:str, args:list, kwargs:dict):
        return 1
    
    urls = {'github': None,
             'website': None,
             'docs': None, 
             'twitter': None,
             'discord': None,
             'telegram': None,
             'linkedin': None,
             'email': None}
    
    def metadata(self, to_string=False, code=False):
        schema = {f:getattr(getattr(self, f), '__metadata__') for f in self.endpoints() if self.is_endpoint(f)}
        metadata = {}
        metadata['schema'] = schema
        metadata['description'] = self.description
        metadata['urls'] = {k: v for k,v in self.urls.items() if v != None}
        if to_string:
            return self.python2str(metadata)
        return metadata
    

    def kwargs2attributes(self, kwargs:dict, ignore_error:bool = False):
        for k,v in kwargs.items():
            if k != 'self': # skip the self
                # we dont want to overwrite existing variables from 
                if not ignore_error: 
                    assert not hasattr(self, k)
                setattr(self, k)

    def num_fns(self):
        return len(self.fns())

    
    def fn2type(self):
        fn2type = {}
        fns = self.fns()
        for f in fns:
            if callable(getattr(self, f)):
                fn2type[f] = self.classify_fn(getattr(self, f))
        return fn2type
    

    @classmethod
    def is_dir_module(cls, path:str) -> bool:
        """
        determine if the path is a module
        """
        filepath = cls.simple2path(path)
        if path.replace('.', '/') + '/' in filepath:
            return True
        if ('modules/' + path.replace('.', '/')) in filepath:
            return True
        return False
    
    @classmethod
    def add_line(cls, path:str, text:str, line=None) -> None:
        # Get the absolute path of the file
        path = cls.resolve_path(path)
        text = str(text)
        # Write the text to the file
        if line != None:
            line=int(line)
            lines = c.get_text(path).split('\n')
            lines = lines[:line] + [text] + lines[line:]
            c.print(lines)

            text = '\n'.join(lines)
        with open(path, 'w') as file:
            file.write(text)


        return {'success': True, 'msg': f'Added line to {path}'}


    @classmethod
    def readme(cls):
        # Markdown input
        markdown_text = "## Hello, *Markdown*!"
        path = cls.filepath().replace('.py', '_docs.md')
        markdown_text =  cls.get_text(path=path)
        return markdown_text
    
    docs = readme


    @staticmethod
    def is_imported(package:str) :
        return  bool(package in sys.modules)
    
    @classmethod
    def is_parent(cls, obj=None):
        obj = obj or cls 
        return bool(obj in cls.get_parents())

    @classmethod
    def find_code_lines(cls,  search:str = None , module=None) -> List[str]:
        module_code = c.module(module).code()
        return c.find_lines(search=search, text=module_code)

    @classmethod
    def find_lines(self, text:str, search:str) -> List[str]:
        """
        Finds the lines in text with search
        """
        found_lines = []
        lines = text.split('\n')
        for line in lines:
            if search in line:
                found_lines += [line]
        
        return found_lines