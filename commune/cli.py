
import commune as c
from munch import Munch

class cli(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 

    def __init__(self, 
                 args = None,
                module = None,
                new_event_loop: bool = True,
                save: bool = True):

        self.get_cli(args=args, 
                     module=module, 
                     new_event_loop=new_event_loop, 
                     save=save)



    def get_cli(
            self,
            args = None,
            module : c.Module = None,
            new_event_loop: bool = True,
            save: bool = True
        ) :
        self.base_module = c.Module()
        input = args or self.argv()
        args, kwargs = self.parse_args(input)
        
        if new_event_loop:
            c.new_event_loop(True)

        if len(args) == 0:
            return c.schema()
        

        base_module_attributes = list(set(self.base_module.functions()  + self.base_module.get_attributes()))
        # is it a fucntion, assume it is for the module
        # handle module/function
        is_fn = args[0] in base_module_attributes

        if '/' in args[0]:
            args = args[0].split('/') + args[1:]
            is_fn = False

        if is_fn:
            # is a function
            module = module or c.Module
            fn = args.pop(0)
        else:
            module = args.pop(0)
            if isinstance(module, str):
                module = c.module(module)
            fn = args.pop(0)
            
        fn_obj = getattr(module, fn)
        
        if callable(fn_obj) :
            if c.classify_fn(fn_obj) == 'self':
                fn_obj = getattr(module(), fn)
            output = fn_obj(*args, **kwargs)
        elif c.is_property(fn_obj):
            output =  getattr(module(), fn)
        else: 
            output = fn_obj  
        if callable(fn):
            output = fn(*args, **kwargs)
        self.process_output(output, save=save)

    def process_output(self, output, save=True, verbose=True):
        if save:
            self.save_history(input, output)
        if c.is_generator(output):
            for output_item in output:
                if isinstance(c, Munch):
                    output_item = output_item.toDict()
                c.print(output_item,  verbose=verbose)
        else:
            if isinstance(output, Munch):
                output = output.toDict()
            c.print(output, verbose=verbose)

    def save_history(self, input, output):
        try:
            self.put(f'cli_history/{int(c.time())}', {'input': input, 'output': output})
        except Exception as e:
            pass
        return {'input': input, 'output': output}
    @classmethod
    def history_paths(cls, n=10):
        return cls.ls('cli_history')[:n]
    
    @classmethod
    def history(cls, n=10):
        history_paths = cls.history_paths(n=n)
        historys = [c.get_json(s) for s in history_paths]
        return historys
    
    @classmethod
    def num_points(cls):
        return len(cls.history_paths())
    
    @classmethod
    def n_history(cls):
        return len(cls.history_paths())

    
    @classmethod
    def clear(cls):
        return cls.rm('cli_history')


        
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
                key, value = arg.split('=', 1)
                # use determine_type to convert the value to its actual type
                
                kwargs[key] = cls.determine_type(value)
            else:
                assert parsing_kwargs is False, 'Cannot mix positional and keyword arguments'
                args.append(cls.determine_type(arg))

        return args, kwargs

    @classmethod
    def determine_type(cls, x):
        if x.lower() == 'null' or x == 'None':
            return None
        elif x.lower() in ['true', 'false']:
            return bool(x.lower() == 'true')
        elif x.startswith('[') and x.endswith(']'):
            # this is a list
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
