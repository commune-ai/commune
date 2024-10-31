import commune as c
import sys
import time
import sys

class cli(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __init__(self, 
                argv = None,
                base = 'module',
                fn_splitters = [':', '/', '//', '::'],
                helper_fns = ['code', 'schema', 'fn_schema', 'help', 'fn_info', 'fn_hash'],
                sep = '--'
                ):
        
        self.argv = self.resolve_argv(argv)
        self.helper_fns = helper_fns
        self.fn_splitters = fn_splitters
        self.sep = sep
        self.base_class = c.module(base)
        self.base_module = self.base_class()
        self.forward(self.argv)

    def forward(self, argv=None):
        t0 = time.time()
        argv = argv or self.argv
        self.input_msg = 'c ' + ' '.join(argv)
        output = None
        init_kwargs = {}
        if any([arg.startswith(self.sep) for arg in argv]): 
            for arg in c.copy(argv):
                if arg.startswith(self.sep):
                    key = arg[len(self.sep):].split('=')[0]
                    if key in self.helper_fns:
                        return self.forward([key , argv[0]])
                    else:
                    
                        value = arg.split('=')[-1] if '=' in arg else True
                        argv.remove(arg)
                        init_kwargs[key] = self.determine_type(value)
        
        # any of the --flags are init kwargs
        fn = argv.pop(0)
        module = self.base_class
        fs = [fs for fs in self.fn_splitters if fs in fn]
        if len(fs) == 1: 
            module, fn = fn.split(fs[0])
            modules = c.modules()
            module_options = []
            for m in modules:
                if module == m:
                    module_options = [m]
                    break
                if module in m:
                    module_options.append(m)
            if len(module_options)>0:
                module = module_options[0]
                module = c.module(module)
            else:
                raise AttributeError(f'Function {fn} not found in {module}')
        if hasattr(module, 'fn2module') and not hasattr(module, fn):
            c.print(f'FN2MODULE ACTIVATED :{fn}')
            fn2module = module.fn2module()
            module = c.module(fn2module[fn])


        fn_obj = getattr(module, fn)
        if c.classify_fn(fn_obj) == 'self':
            fn_obj = getattr(module(**init_kwargs), fn)
        if callable(fn_obj):
            args, kwargs  = self.parse_args(argv)
            output = fn_obj(*args, **kwargs)
        else:
            output = fn_obj
        buffer = '⚡️'*4
        c.print(buffer+fn+buffer, color='yellow')
        latency = time.time() - t0
        is_error =  c.is_error(output)
        if is_error:
            msg =  f'❌Error({latency:.3f}sec)❌' 
        else:
            msg = f'✅Result({latency:.3f}s)✅'
        c.print(msg)
        is_generator = c.is_generator(output)
        if is_generator:
            # print the items side by side instead of vertically
            for item in output:
                if isinstance(item, dict):
                    c.print(item)
                else:
                    c.print(item, end='')
        else:
            c.print(output)
        return output
    
    def is_property(self, obj):
        return isinstance(obj, property)

    def parse_args(self, argv = None):
        argv = argv or self.argv
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = self.determine_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(self.determine_type(arg))
        return args, kwargs

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
            try:
                return int(x)
            except ValueError:
                try:
                    return float(x)
                except ValueError:
                    return x

    def resolve_argv(self, argv):
        argv = argv or sys.argv[1:]
        return argv

          
def main():
    cli()