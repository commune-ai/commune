
import commune as c
import json
import sys
import time
import os
import threading
import sys


class cli:
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 

    def __init__(self, 
                 args = None,
                module = 'module',
                verbose = True,
                forget_fns = ['module.key_info', 'module.save_keys'], 
                seperator = ' ',
                buffer_size=4,
                helper_fns = ['code', 'schema', 'fn_schema', 'help', 'fn_info'],
                save: bool = False):
        
        self.helper_fns = helper_fns
        self.seperator = seperator
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.save = save
        self.forget_fns = forget_fns
        self.base_module = c.module(module) if isinstance(module, str) else module
        self.base_module_attributes = list(set(self.base_module.functions()  + self.base_module.attributes()))

        self.forward(args)

    def forward(self, argv=None):

        t0 = time.time()
        argv = argv or self.argv()
        self.input_msg = 'c ' + ' '.join(argv)
        output = None

        init_kwargs = {}
        if any([arg.startswith('--') for arg in argv]): 
            for arg in c.copy(argv):
                if arg.startswith('--'):
                    key = arg[2:].split('=')[0]
                    if key in self.helper_fns:
                        new_argvs = self.argv()
                        new_argvs.remove(arg)
                        new_argvs = [key , new_argvs[0]]
                        return self.forward(new_argvs)
                    argv.remove(arg)
                    if '=' not in arg:
                        value = True
                    value = arg.split('=')[1]
                    init_kwargs[key] = self.determine_type(value)
        
        # any of the --flags are init kwargs
        if argv[0].endswith('.py'):
            argv[0] = argv[0][:-3]
        

        if ':' in argv[0]:
            # {module}:{fn} arg1 arg2 arg3 ... argn
            argv[0] = argv[0].replace(':', '/')

        if '/' in argv[0]:
            # prioritize the module over the function
            module = '.'.join(argv[0].split('/')[:-1])
            fn = argv[0].split('/')[-1]
            argv = [module , fn , *argv[1:]]
            is_fn = False
        else:
            is_fn = argv[0] in self.base_module_attributes 

        if is_fn:
            module = self.base_module
            fn = argv.pop(0)
        else:
            module = argv.pop(0)
            fn = argv.pop(0)

        if isinstance(module, str):
            module = c.get_module(module)
        # module = self.base_module.from_object(module)
        module_name = module.module_name()
        fn_path = f'{module_name}/{fn}'
        fn_obj = getattr(module, fn)
        fn_type = c.classify_fn(fn_obj)
        if fn_type == 'self' or len(init_kwargs) > 0:
            fn_obj = getattr(module(**init_kwargs), fn)
        # calling function buffer
        input_msg = f'[bold]fn[/bold]: {fn_path}'

        if callable(fn_obj) and not c.is_property(fn_obj):
            args, kwargs  = self.parse_args(argv)
            if len(args) > 0 or len(kwargs) > 0:
                inputs = {"args":args, "kwargs":kwargs}
                input_msg += ' ' + f'[purple][bold]params:[/bold] {json.dumps(inputs)}[/purple]'
            output = lambda: fn_obj(*args, **kwargs)
        else: 
            output = lambda: fn_obj 
        self.input_msg = input_msg
        buffer = '⚡️'*4
        c.print(buffer+input_msg+buffer, color='yellow')
        output =  output()
        latency = time.time() - t0
        is_error =  c.is_error(output)

        if is_error:
            buffer = '❌'
            msg =  f'Error(latency={latency:.3f})' 
        else:
            buffer = '✅'
            msg = f'Result(latency={latency:.3f})'

        print(buffer + msg + buffer)
        
        num_spacers = max(0,  len(self.input_msg) - len(msg) )
        left_spacers = num_spacers//2
        right_spacers = num_spacers - left_spacers
        msg = self.seperator*left_spacers + msg + self.seperator*right_spacers
        buffer =  self.buffer_size * buffer
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