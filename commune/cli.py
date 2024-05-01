
import commune as c
from munch import Munch

class cli(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 

    def __init__(self, 
                 args = None,
                module = 'module',
                new_event_loop: bool = True,
                verbose = True,
                save: bool = True):
        self.base_module = c.module(module)
        args = args or self.argv()
        output = self.get_output(args)

        if c.is_generator(output):
            for output_item in output:
                if isinstance(c, Munch):
                    output_item = output_item.toDict()
                c.print(output_item,  verbose=verbose)
        else:
            if isinstance(output, Munch):
                output = output.toDict()
            c.print(output, verbose=verbose)
        
        if save:
            self.history_module().add({'input': 'c ' + ' '.join(args), 'output': output})

    def get_output(self, args):

        args, kwargs = self.parse_args(args)

        base_module_attributes = list(set(self.base_module.functions()  + self.base_module.get_attributes()))
        # is it a fucntion, assume it is for the module
        # handle module/function
        is_fn = args[0] in base_module_attributes

        if '/' in args[0]:
            args = args[0].split('/') + args[1:]
            is_fn = False

        if is_fn:
            # is a function
            module = self.base_module
            fn = args.pop(0)
        else:
            module = args.pop(0)
            if isinstance(module, str):
                module = c.module(module)
            fn = args.pop(0)


        if module.classify_fn(fn) == 'self':
            module = module()
            
        fn_obj = getattr(module, fn)

        if callable(fn_obj):
            output = fn_obj(*args, **kwargs)
        elif c.is_property(fn_obj):
            output =  getattr(module(), fn)
        else: 
            output = fn_obj  
        if callable(fn):
            output = fn(*args, **kwargs)


        return output


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

        if x.startswith('py(') and x.endswith(')'):
            try:
                return eval(x[3:-1])
            except:
                return x
        if x.lower() in 'null' or x == 'None':  # convert 'null' or 'None' to None
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
                

    @classmethod
    def history_module(cls, path='history'):
        return c.m('history')( cls.resolve_path(path))


    @classmethod
    def history(cls,**kwargs):
        history = cls.history_module().history(**kwargs)
        return history
    

    @classmethod
    def history_paths(cls, **kwargs):
        history = cls.history_module().history_paths(**kwargs) 
        return history

def main():
    cli()
