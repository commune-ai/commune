import sys
import time
import sys
import commune as c
print = c.print
class Cli:

    def forward(self):
        argv = sys.argv[1:]
        if len(argv) == 0:
            argv = ['vs']
        fn = self.get_fn(argv)
        args, kwargs = self.get_args_kwargs(argv)
        return self.run_fn(fn, args, kwargs)

    def determine_type(self, x):
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
                x =  [self.determine_type(item.strip()) for item in list_items]
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
                return {key.strip(): self.determine_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
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

    def get_args_kwargs(self, argv):
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in c.copy(argv):
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = self.determine_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(self.determine_type(arg))
        return args, kwargs


    def get_init_kwargs(self, argv:list, helper_fns:list = ['code', 'schema', 'fn_schema', 'help', 'fn_info', 'fn_hash']):
        init_kwargs = {}
        for arg in c.copy(argv):
            if arg.startswith('--'): # init kwargs
                key = arg[len('--'):].split('=')[0]
                if key in helper_fns:
                    # is it a helper function
                    return self.forward([key , argv[0]])
                else:
                    value = arg.split('=')[-1] if '=' in arg else True
                    argv.remove(arg)
                    init_kwargs[key] = self.determine_type(value)
                continue
        return init_kwargs

    def get_fn(self, argv:list, init_kwargs:dict={}, default_fn:str='forward', default_module:str='module'):
        if len(argv) == 0:
            fn = default_fn
        else:
            fn = argv.pop(0).replace('-', '_')
        init_kwargs = self.get_init_kwargs(argv)
        # get the function object
        if  '/' in fn and '::' in fn:
            fn_splitter = '::'
            module = fn.split(fn_splitter)[0]
            fn = fn.split(fn_splitter)[-1]
        elif '/' in fn:
            fn_splitter = '/'
            module = fn_splitter.join(fn.split(fn_splitter)[:-1])
            fn = fn.split(fn_splitter)[-1]
        else:
            module = default_module


        module = c.shortcuts.get(module, module)
        print('⚡️'*4+f'{module}::{fn}'+'⚡️'*4, color='yellow')
        module = c.module(module)
        if not hasattr(module, fn):
            return {'error': f'module::{fn} does not exist', 'success': False}
        fn_obj = getattr(module, fn)
        initialize_module_class = isinstance(fn, property) or 'self' in c.get_args(fn_obj)
        module = module(**init_kwargs) if initialize_module_class else module
        fn_obj = getattr(module, fn)
        return fn_obj

    def run_fn(self, fn_obj, args, kwargs):
        # call the function
        t0 = time.time()
        output = fn_obj(*args, **kwargs) if callable(fn_obj) else fn_obj
        latency = time.time() - t0
        is_error =  c.is_error(output)
        print(f'❌Error({latency:.3f}sec)❌' if is_error else f'🟢Result({latency:.3f}s)🟢')
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
