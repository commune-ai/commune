
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
            for i in output:
                if isinstance(c, Munch):
                    i = i.toDict()
                c.print(i,  verbose=verbose)
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


        