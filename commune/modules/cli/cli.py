
import commune as c
from munch import Munch

class CLI(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 
    def __init__(
            self,
            module = 'module', 
            fn=  None,
            new_event_loop: bool = True,
            save: bool = True


        ) :
        self.module = c.Module()
        input = self.argv()
        args, kwargs = self.parse_args(input)
        
        module_list = c.modules()
        if new_event_loop:
            c.new_event_loop(True)

        if len(args) == 0:
            output = c.schema()
        elif len(args)> 0:
            functions = list(set(self.module.functions()  + self.module.get_attributes()))
            # is it a fucntion, assume it is for the module
            module_list = c.modules()
            # handle module/function
            is_fn = args[0] in functions

            if '/' in args[0]:
                args = args[0].split('/') + args[1:]
                is_fn = False

            is_module = bool(not is_fn)
            if is_fn:
                # is a function
                module = c.Module
                fn = args.pop(0)
            elif is_module:
                module = args.pop(0)
                module = c.module(module)
            
            else:
                # is a a namespace
                namespace = self.namespace(update=False)
                if args[0] in namespace:
                    module = args.pop(0)
                    module = c.connect(module)
                else: 
                    raise Exception(f'No module, function or server found for {args[0]}')
            
            if fn == None:
                if len(args) == 0:
                    fn = "__init__"
                else: 
                    fn = args.pop(0)
                    
                    
            if fn != '__init__':
                fn_name = fn
                fn = getattr(module, fn_name)
                

                # if c.is_property(fn):
                #     output = getattr(module(), fn.__name__)
                
                if callable(fn) :
                    if c.classify_fn(fn) == 'self':
                        module_inst = module()
                        fn = getattr(module_inst, fn_name)
                elif c.is_property(fn):
                    output =  getattr(module(), fn_name)
                else: 
                    output = fn    
                
            else:
                fn = module
            if callable(fn):
                output = fn(*args, **kwargs)
            
                
        else:
            raise Exception ('No module, function or server found for {args[0]}')

        if save:
            self.save_history(input, output)
 
        if c.is_generator(output):
            for i in output:
                if isinstance(c, Munch):
                    i = i.toDict()
                c.print(i)
        else:
            if isinstance(output, Munch):
                output = output.toDict()
            c.print(output)

    def save_history(self, input, output):
        try:
            self.put(f'cli_history/{int(c.time())}', {'input': input, 'output': output})
        except Exception as e:
            pass
        return {'input': input, 'output': output}
    @classmethod
    def history_paths(cls):
        return cls.ls('cli_history')
    
    def history(self):
        history_paths = self.history_paths()
        historys = [c.get_json(path) for path in history_paths]
        return historys
    
    @classmethod
    def clear(cls):
        return cls.rm('cli_history')


        