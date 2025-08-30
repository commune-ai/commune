import sys
import time
import sys
from typing import Any
import commune as c
import inspect

class Cli:

    def main(self,
                fn='vs',  
                module='module', 
                default_fn = 'forward'):
        t0 = time.time()
        argv = sys.argv[1:]
        # ---- FUNCTION
        module = self.module(module)()
        if len(argv) == 0:
            argv += [fn]

        fn = argv.pop(0)

        if hasattr(module, fn):
            fn_obj = getattr(module, fn)
        elif '/' in fn:
            if fn.startswith('/'):
                fn = fn[1:]
            if fn.endswith('/'):
                fn = fn + default_fn
            new_module = '/'.join(fn.split('/')[:-1]).replace('/', '.')
            module =  self.module(new_module)()
            fn = fn.split('/')[-1]
            fn_obj = getattr(module, fn)

        else:
            raise Exception(f'Function {fn} not found in module {module}')
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
        result = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        speed = time.time() - t0
        module_name = module.__class__.__name__
        self.print(f'Call({module_name}/{fn}, speed={speed:.2f}s)')
        duration = time.time() - t0
        is_generator = self.is_generator(result)
        if is_generator:
            for item in result:
                if isinstance(item, dict):
                    self.print(item)
                else:
                    self.print(item, end='')
        else:
            self.print(result)


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


    def print(self, *args, **kwargs):
        """
        Print with a custom prefix
        """
        prefix = kwargs.pop('prefix', '')
        if prefix:
            print(f'{prefix}: ', end='')
        print(*args, **kwargs)


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
