import commune as c
import sys
import time
import sys
print = c.print
def determine_type(x):
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
            x =  [determine_type(item.strip()) for item in list_items]
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
            return {key.strip(): determine_type(value.strip()) for key, value in [item.split(':', 1) for item in dict_items]}
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
def forward(argv = None,
            sep = '--', 
            fn_splitters = [':', '/', '//', '::'],
            base = 'module', 
            helper_fns = ['code', 'schema', 'fn_schema', 'help', 'fn_info', 'fn_hash'], 
            default_fn = 'vs'):
    t0 = time.time()
    argv = argv or sys.argv[1:]
    if len(argv) == 0:
        argv = [default_fn]
    output = None
    init_kwargs = {}
    if any([arg.startswith(sep) for arg in argv]): 
        for arg in c.copy(argv):
            if arg.startswith(sep):
                key = arg[len(sep):].split('=')[0]
                if key in helper_fns:
                    # is it a helper function
                    return forward([key , argv[0]])
                else:
                    value = arg.split('=')[-1] if '=' in arg else True
                    argv.remove(arg)
                    init_kwargs[key] = determine_type(value)
    # any of the --flags are init kwargs
    fn = argv.pop(0).replace('-', '_')
    module = c.module(base)
    fs = [fs for fs in fn_splitters if fs in fn]
    if len(fs) == 1: 
        module, fn = fn.split(fs[0])
        module = c.shortcuts.get(module, module)
        modules = c.get_modules()
        module_options = []
        for m in modules:
            if module == m:
                module_options = [m]
                break
            if module in m:
                module_options.append(m)
        if len(module_options)>0:
            module = module_options[0]
            print('Module:', module)
            module = c.module(module)
        else:
            raise AttributeError(f'Function {fn} not found in {module}')
    if hasattr(module, 'fn2module') and not hasattr(module, fn):
        c.print(f'ROUTE_ACTIVATED({fn} from {module})')
        fn2module = module.fn2module()
        if not fn in fn2module:
            return c.print(f'FN({fn}) not found {module}', color='red')
        module = c.module(fn2module[fn])
    fn_obj = getattr(module, fn)
    if c.is_property(fn_obj) or c.classify_fn(fn_obj) == 'self':
        fn_obj = getattr(module(**init_kwargs), fn)
    if callable(fn_obj):
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = determine_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(determine_type(arg))
        output = fn_obj(*args, **kwargs)
    else:
        output = fn_obj
    buffer = '⚡️'*4
    c.print(buffer+fn+buffer, color='yellow')
    latency = time.time() - t0
    is_error =  c.is_error(output)
    msg =  f'❌Error({latency:.3f}sec)❌' if is_error else f'✅Result({latency:.3f}s)✅'
    c.print(msg)
    is_generator = c.is_generator(output)
    if is_generator:
        for item in output:
            if isinstance(item, dict):
                c.print(item)
            else:
                c.print(item, end='')
    else:
        c.print(output)
    return output
def main():
    forward()