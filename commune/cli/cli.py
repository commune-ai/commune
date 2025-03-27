import sys
import time
import sys
from typing import Any
import commune as c
print = c.print

class Cli:
    desc = 'cli for running functions'
    safety = False
    ai_enabled = False

    def get_params(self, argv):
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = self.str2python(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(self.str2python(arg))
        return {'args': args, 'kwargs': kwargs}

    def get_fn(self, 
                argv:list, 
                default_fn:str='forward',
                splitter='/', 
                default_module:str='module', 
                helper_fns=['code']):
        if len(argv) == 0:
            fn = default_fn
        else:
            fn = argv.pop(0).replace('-', '_')
        if fn.endswith('/'):
            fn += 'forward'
        # get the function object
        if splitter in fn:
            module = splitter.join(fn.split(splitter)[:-1])
            module = module.replace(splitter, '.')
            fn = fn.split(splitter)[-1]
        else:
            module = default_module
        print(f'Call({module}/{fn})', color='yellow')
        module = c.module(module)()
        return getattr(module, fn)

    def forward(self):
        if self.safety:
            assert c.pwd() != c.home_path, 'Cannot run cli in home directory, please change if you want to run'
        t0 = time.time()
        argv = sys.argv[1:]
        if len(argv) == 0:
            fn_obj = c.vs
            params = {'args': [], 'kwargs': {}}
        else:
            fn_obj = self.get_fn(argv) # get the function object
            params = self.get_params(argv) # get the parameters
        output = fn_obj(*params['args'], **params['kwargs']) if callable(fn_obj) else fn_obj
        if self.ai_enabled:
            output = c.ask(output)
        duration = time.time() - t0
        print(f'❌Error({duration:.3f}sec)❌' if c.is_error(output) else f'✅Result({duration:.3f}s)✅')
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