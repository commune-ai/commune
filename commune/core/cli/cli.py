import sys
import time
import sys
from typing import Any
import inspect
import commune as c

class Cli:

    def get_argv(self, argv=None):
        """
        Get the arguments passed to the script
        """
        argv = argv or sys.argv[1:] # remove the first argument (the script name)
        return argv



    def forward(self, module='module', fn='forward', argv=None, **kwargs):

        t0 = time.time()
        argv = self.get_argv(argv)
       
        # ---- MODULE/FN ----
        fn_obj = None

        base_module = module
        module_obj = c.module(module)()
        if len(argv) == 0:
            # scenario 1: no arguments, use the default function
            fn = 'vs'
        elif len(argv) > 0 and hasattr(module_obj, argv[0]):
            # scenario 2: first argument is the function name c 
            fn = argv.pop(0)
        elif c.module_exists(argv[0]):
            # scenario 3: the fn name is of another module so we will look it up in the fn2module
            # and then get the function from the module
            module = argv.pop(0)
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
            assert fn in fn2module, f'Function {fn} not found in module {module}'
            module = fn2module[fn]

        if module != base_module:
            module_obj = c.module(module)()
        if not hasattr(module_obj, fn):
            print(f'Function {fn} not found in module {module}')
            argv.insert(0, module) # this is a hack to ensure c api/code --> c code api assuming code is not in the api module
            module = base_module

        module_obj = c.module(module)()
        fn_obj = getattr(module_obj, fn)
        # ---- PARAMS ----
        params = self.get_fn_params(argv)
        c.print(f'Request(module={module} fn={fn} params={params})')


        # ---- RESULT ----
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        response_time = time.time() - t0
        c.print(f'Result(speed={response_time:.2f}s)')
        duration = time.time() - t0
        is_generator = self.is_generator(result)
        if is_generator:
            for item in result:
                if isinstance(item, dict):
                    c.print(item)
                else:
                    c.print(item, end='')
        else:
            c.print(result)

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
                x =  [str2python(item.strip()) for item in list_items]
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
                return {key.strip(): str2python(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
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

    def shorten(self, x, max_length=12):
        """
        Shorten the hash to 8 characters
        """
        return x[:max_length] + '...' + x[-max_length:]

    def get_fn_params(self, argv):

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
        return params

    def get_module_init_params(self,argv = None):

        # anything that has --{arg}={value} in the args
        # will be considered as a parameter to the __init__ function of the module before the function is loaded
        argv = self.get_argv(argv)
        params  = {}
        for arg in argv:
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' in arg:
                    key, value = arg.split('=')
                    params[key] = self.str2python(value)
                else:
                    raise Exception(f'Invalid argument {arg}')

        return params
