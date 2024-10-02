
import commune as c
import sys
import time
import sys


class cli:
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 

    def __init__(self, 
                args = None,
                base = 'module',
                fn_splitters = [':', '/', '//', '::'],
                helper_fns = ['code', 'schema', 'fn_schema', 'help', 'fn_info'],
                sep = '--'):
        
        self.helper_fns = helper_fns
        self.fn_splitters = fn_splitters
        self.sep = sep
        self.base_module = c.module(base)()
        self.forward(args)

    def forward(self, argv=None):
        t0 = time.time()
        argv = argv or self.argv()
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
                        value = arg.split('=')[-1] if '=' in arg else value
                        argv.remove(arg)
                        init_kwargs[key] = self.determine_type(value)
        
        # any of the --flags are init kwargs
        fn = argv.pop(0)
        base_attributes = dir(self.base_module)
        if fn in base_attributes:
            fn_obj = getattr(self.base_module, fn)
        else: 
            fs = [fs for fs in self.fn_splitters if fs in fn]
            assert len(fs) == 1, f'Function splitter not found in {fn}'
            print(fn)
            module, fn = fn.split(fs[0])

            if c.module_exists(module):
                module = c.get_module(module)
                fn_obj = getattr(module, fn)
                if c.classify_fn(fn_obj) == 'self':
                    fn_obj = getattr(module(**init_kwargs), fn)
            else:
                raise AttributeError(f'Function {fn} not found in {module}')
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
            buffer = '✅'
            msg = f'✅Result({latency:.3f}s)✅'
        print( msg )
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
    
        # c.print( f'Result ✅ (latency={self.latency:.2f}) seconds ✅')
    


    @classmethod
    def is_property(cls, obj):
        return isinstance(obj, property)
        


    @classmethod
    def parse_args(cls, argv = None):
        if argv is None:
            argv = cls.argv()
        args = []
        kwargs = {}
        parsing_kwargs = False
        for arg in argv:
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                kwargs[key] = cls.determine_type(value)

            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(cls.determine_type(arg))
        return args, kwargs

    @classmethod
    def determine_type(cls, x):
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
    

    def argv(self):
        return sys.argv[1:]
          
def main():
    cli()