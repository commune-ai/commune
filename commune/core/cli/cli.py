import sys
import time
import sys
from typing import Any
import inspect
import commune as c

class Cli(c.Module):

    def forward(self,
                module='module', 
                fn='forward'):

        t0 = time.time()
        argv = sys.argv[1:] # remove the first argument (the script name)
        module_obj = c.module(module)()
        fn_obj = None
        if len(argv) == 0:
            # scenario 1: no arguments, use the default function
            fn_obj = getattr(module_obj, fn)
        elif len(argv) > 0 and hasattr(module_obj, argv[0]):
            # scenario 2: first argument is the function name c 
            fn_obj = getattr(module_obj, argv.pop(0))
        elif len(argv) >= 2 and c.module_exists(argv[0]):
            # scenario 3: first argument is the module name c module fn *args **kwargs
            module_obj = c.module(argv.pop(0))()
            if hasattr(module_obj, argv[0]):
                fn_obj = getattr(module_obj, argv.pop(0))
            else:
                raise Exception(f'Function {argv[0]} not found in module {module}')
        elif len(argv[0].split('/')) == 2:
            # scenario 4: first argument is a path to a function c module/fn *args **kwargs
            fn = argv.pop(0)
            module_obj =  c.module(fn.split('/')[0])()
            fn_obj = getattr(module_obj, fn.split('/')[1])
        else:
            fn = argv.pop(0)
            fn2module = self.fn2module()
            if fn in fn2module:
                module = fn2module[fn]
                module_obj = c.module(fn2module[fn])()
                print(f'fn2module({fn} -> {module}/{fn})')
                fn_obj = getattr(module_obj, fn)
            else:
                raise Exception(f'Function {fn} not found in module {module}')
        assert fn_obj is not None, f'Function {fn} not found in module {module}'
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
        # run thefunction
        module_name = module.__class__.__name__.lower()

        params_hash = self.shorten(c.hash(params))
        c.print(f'Request(module={module_name} fn={fn} tx_hash={params_hash})')
        print(fn_obj)
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        speed = time.time() - t0



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