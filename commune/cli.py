
import commune as c
import json

import sys
import time
import threading

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
                save: bool = False):
        self.t0 = time.time()

        self.verbose = verbose
        self.save = save
        self.forget_fns = forget_fns
        
        self.base_module = c.module(module)() if isinstance(module, str) else module
        self.base_module_attributes = list(set(self.base_module.functions()  + self.base_module.attributes()))
        self.forward(args=args)
    
    def forward(self, args=None):
        args = args or self.argv()
        self.input_str = 'c ' + ' '.join(args)
        output = self.get_output(args)
        output = self.process_output(output)

        self.latency = time.time() - self.t0
        # c.print( f'Result ✅ (latency={self.latency:.2f}) seconds ✅')
    
    def argv(self):
        import sys
        return sys.argv[1:] 


    def process_output(self, output):

        is_generator = self.base_module.is_generator(output)
        if self.base_module.is_generator(output):
            for output_item in output:
                c.print(output_item)
        else:
            c.print(output)
        return output
    


    def get_output(self, argv):

        """
        the cli works as follows 
        c {module}/{fn} arg1 arg2 arg3 ... argn
        if you are calling a function ont he module function (the root module), it is not necessary to specify the module
        c {fn} arg1 arg2 arg3 ... argn
        """

        if ':' in argv[0]:
            # {module}:{fn} arg1 arg2 arg3 ... argn
            argv = argv[0].split(':') + argv[1:]
            is_fn = False
        elif '/' in argv[0]:
            argv = argv[0].split('/') + argv[1:]
            is_fn = False
        else:
            is_fn = argv[0] in self.base_module_attributes
    
        if is_fn:
            module = self.base_module
            fn = argv.pop(0)
        else:
            module = argv.pop(0)
            if isinstance(module, str):
                module = c.module(module)
            fn = argv.pop(0)
        

        # module = self.base_module.from_object(module)


        fn_class = module.classify_fn(fn) if hasattr(module, 'classify_fn') else self.base_module.classify_fn(fn)

        if fn_class == 'self':
            if callable(module):
                module = module()
        module_name = module.module_path()
        fn_path = f'{module_name}/{fn}'
        fn_obj = getattr(module, fn)

        if callable(fn_obj):

            args, kwargs = self.parse_args(argv)

            # spinning status
            inputs = json.dumps({"args":args, "kwargs":kwargs})
            msg = f'Calling Function -> {fn_path}/{inputs})'
            output = lambda: fn_obj(*args, **kwargs)
            color = 'blue'
            emoji = '🔵'
        elif self.is_property(fn_obj):
            inputs = ''
            msg = f'Calling Function Property --> {fn_path})'
            output =  lambda : getattr(module(), fn)
            color = 'green'
            emoji = '🟢'
        else: 
            msg = f'Calling Attribute {fn_path}'
            output = lambda: fn_obj 
            color = 'yellow'
            emoji = '🟡'
        emoji = emoji * 3
        c.print('[b]' + emoji + msg + emoji + '\b' , color=color)
        return output()
    

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
            # TODO fix exception with  "="
            # if any([arg.startswith(_) for _ in ['"', "'"]]):
            #     assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
            #     args.append(cls.determine_type(arg))
            if '=' in arg:
                parsing_kwargs = True
                key, value = arg.split('=')
                # use determine_type to convert the value to its actual type
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
                
def main():
    cli()