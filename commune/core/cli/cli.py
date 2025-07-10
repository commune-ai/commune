import sys
import time
import sys
from typing import Any
import inspect
import commune as c
from typing import List
from copy import deepcopy
import json

class Cli:
    def __init__(self, 
                key=None, 
                default_fn='go',
                ):

        self.default_fn = default_fn

    def forward(self, module='module', fn='forward', argv=None, **kwargs):
        """
        Forward the function to the module and function
        
        """

        time_start = time.time()

        # ---- MODULE/FN ----\
        time_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_start))
        c.print(f'[{time_string}] Forwarding {module}/{fn} with args: {argv}', color='blue')
        argv = self.get_argv(argv)
        argv, module, fn = self.get_module_fn(module, fn, argv)
        argv, init_params = self.get_init_params(argv)
        argv, params = self.get_fn_params(argv)
        fn_obj = getattr(c.mod(module)(**init_params), fn)
        schema = self.get_schema(module, fn)
        c.print(f'Calling({module}/{fn})')
        result = fn_obj(*params["args"], **params["kwargs"]) if callable(fn_obj) else fn_obj
        duration = round(time.time() - time_start, 3)
        c.print(f'Result(duration={duration}s)\n')
        self.print_result(result)

    def print_result(self, result:Any):
        is_generator = self.is_generator(result)
        if is_generator:
            
            for item in result:
                if isinstance(item, dict):
                    c.print(item)
                else:
                    c.print(item, end='')
        else:
            c.print(result)

    def get_schema(self, module, fn, verbose=False):
        try:
            schema = c.schema(module + '/' + fn)
        except Exception as e:
            if verbose:
                c.print(f'Error getting schema for {module}/{fn}: {e}', color='red')
            schema = {}
        return  schema
    def get_params(self, params:dict, schema:dict):
        if 'args' and 'kwargs' in params:
            args = params['args']
            kwargs = params['kwargs']
        params = {}
        input_schema = schema.get('input', {})
        input_schema_keys = list(input_schema.keys())
        for i, arg in enumerate(args):
            if i < len(input_schema):
                params[input_schema_keys[i]] = arg
        params = {**params, **kwargs}  # merge args and kwargs
        return params


    def shorten(self, x:str, n=12):
        if len(x) > n:
            return x[:n] +  '...' + x[-n:]
        return x

    
    def get_argv(self, argv=None):
        """
        Get the arguments passed to the script
        """
        argv = argv or sys.argv[1:] # remove the first argument (the script name)
        return argv


    def is_generator(self, obj):
        """
        Is this shiz a generator dawg?
        """
        if isinstance(obj, str):
            if not hasattr(self, obj):
                return False
            obj = getattr(self, obj)
        if not callable(obj):
            result = inspect.isgenerator(obj)
        else:
            result =  inspect.isgeneratorfunction(obj)
        return result


    def get_fn_params(self, argv) -> tuple:

        # ---- PARAMS ----
        params = {'args': [], 'kwargs': {}} 
        parsing_kwargs = False
        if len(argv) > 0:
            for arg in argv:
                if '=' in arg:
                    parsing_kwargs = True
                    key, value = arg.split('=')
                    params['kwargs'][key] = self.str2python(value)
                else:
                    assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                    params['args'].append(self.str2python(arg))        
        return argv, params


    def get_module_fn(self, module:str, fn:str, argv):
        
        module_obj = c.mod(module)()
        if len(argv) == 0:
            # scenario 1: no arguments, use the default function
            fn = self.default_fn
        elif len(argv) > 0 and hasattr(module_obj, argv[0]):
            fn = argv.pop(0)
        elif argv[0].endswith('/'):
            # scenario 4: the fn name is of another module so we will look it up in the fn2module
            module = argv.pop(0)[:-1]
        elif argv[0].startswith('/'):
            # scenario 5: the fn name is of another module so we will look it up in the fn2module
            fn = argv.pop(0)[1:]
        elif len(argv[0].split('/')) == 2:
            # scenario 6: first argument is a path to a function c module/fn *args **kwargs
            module, fn = argv.pop(0).split('/')
        elif len(argv) >= 2 and c.module_exists(argv[0]):
            # scenario 7: first argument is the module name c module fn *args **kwargs
            module = argv.pop(0)
            fn = argv.pop(0)
        else:
            # scenario 8: the fn name is of another module so we will look it up in the fn2module
            # and then get the function from the module
            fn = argv.pop(0)
            fn2module = c.fn2module()
            if not fn in fn2module:
                raise ValueError(f'Function {fn} not found in module {module}')
            module = fn2module[fn]
        return argv, module, fn

    def get_params_from_args(self, args:List[str],kwargs:List[str], schema:dict):
        """
        Get the parameters from the args and kwargs
        """
        params = {}
        for i, arg in enumerate(args):
            if i < len(schema):
                params[schema[i]] = self.str2python(arg)
            else:
                params[f'arg{i}'] = self.str2python(arg)
        for key, value in kwargs.items():
            if key in schema:
                params[key] = self.str2python(value)
            else:
                params[key] = self.str2python(value)
        return params

    def str2python(self, x):
        x = str(x)
        if isinstance(x, str) :
            if x.startswith('py(') and x.endswith(')'):
                try:
                    return eval(x[3:-1])
                except:
                    return x
        if x.lower() in ['null'] or x == 'None':  # convert 'null' or 'None' to None
            return None 
        elif x.lower() in ['true', 'false']: # convert 'true' or 'false' to bool
            return bool(x.lower() == 'true')
        elif x.startswith('[') and x.endswith(']'): # this is a list
            try:
                list_items = x[1:-1].split(',')
                # try to convert each item to its actual type
                x =  [self.str2python(item.strip()) for item in list_items]
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
                return {key.strip(): self.str2python(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
            except:
                # if conversion fails, return as string
                return x
        else:
            # try to convert to int or float, otherwise return as string
            
            for type_fn in [int, float]:
                try:
                    return type_fn(x)
                except ValueError:
                    pass
        return x

    def get_init_params(self, argv) -> tuple:
        """
        Get the initial parameters from the arguments
        """
        new_argv = []
        init_params = {}
        for i, arg in enumerate(argv):
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' in arg:
                    key, value = arg.split('=')
                    init_params[key] = self.str2python(value)
                else:
                    init_params[arg] = True
            else:
                new_argv.append(arg)
        return new_argv, init_params
