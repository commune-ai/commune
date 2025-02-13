import sys
import time
import sys
import commune as c
from typing import Any
print = c.print

class Cli:
    desc = 'commune cli for running functions'
    def __init__(self):
        self.forward()
    def forward(self):
        t0 = time.time()
        argv = sys.argv[1:]
        if len(argv) == 0:
            argv = ['vs']
        fn_obj = self.get_fn(argv) # get the function object
        params = self.get_params(argv) # get the parameters
        output = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        latency = time.time() - t0
        print(f'❌Error({latency:.3f}sec)❌' if self.is_error(output) else f'✅Result({latency:.3f}s)✅')
        is_generator = c.is_generator(output)
        if is_generator:
            for item in output:
                if isinstance(item, dict):
                    print(item)
                else:
                    print(item, end='')
        else:
            print(output)
        return output

    def parse_type(self, x):
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
                x =  [self.parse_type(item.strip()) for item in list_items]
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
                return {key.strip(): self.parse_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
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

    def get_params(self, argv):
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = self.parse_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(self.parse_type(arg))
        return {'args': args, 'kwargs': kwargs}


    def get_fn(self, argv:list, init_kwargs:dict={}, default_fn:str='forward', default_module:str='module', helper_fns=['code']):
        if len(argv) == 0:
            fn = default_fn
        else:
            fn = argv.pop(0).replace('-', '_')

        init_kwargs = {}
        for arg in argv:
            if arg.startswith('--'): # init kwargs
                k = arg[len('--'):].split('=')[0]
                if k in helper_fns: 
                    return self.forward([k , argv[0]])
                else:
                    v = arg.split('=')[-1] if '=' in arg else True
                    argv.remove(arg)
                    init_kwargs[key] = self.parse_type(v)
                continue
        if fn.endswith('/'):
            fn += 'forward'
        # get the function object
        for splitter in ['::', '/']:
            if splitter in fn:
                module = splitter.join(fn.split(splitter)[:-1])
                module = module.replace(splitter, '.')
                fn = fn.split(splitter)[-1]
                break
        else:
            module = default_module
        if module.endswith('.py'):
            module = module[:-3]
        shortcut = False
        if module in c.shortcuts:
            old_module = module
            module = c.shortcuts[module]
            shortcut = True
            print(f'ShortcutEnabled({old_module} -> {module})', color='yellow')

        filepath = c.filepath(module).replace(c.home_path, '~')    
        print(f'Call({module}/{fn}, path={filepath} shortcut={shortcut})', color='yellow')

        module = c.module(module)
        fn_obj = getattr(module, fn)
        if isinstance(fn_obj, property) or 'self' in c.get_args(fn_obj):
            module = module(**init_kwargs)
        if not hasattr(module, fn):
            module = module()
            if not hasattr(module, fn):
                return {'error': f'module/{fn} does not exist', 'success': False}
        fn_args = c.get_args(fn_obj)
        # initialize the module class if it is a property or if 'self' is in the arguments
        return getattr(module, fn)

    
    @staticmethod
    def is_error( x:Any):
        """
        The function checks if the result is an error
        The error is a dictionary with an error key set to True
        """
        if isinstance(x, dict):
            if 'error' in x and x['error'] == True:
                return True
            if 'success' in x and x['success'] == False:
                return True
        return False
