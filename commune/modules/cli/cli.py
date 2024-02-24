
import commune as c


class CLI(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 
    def __init__(
            self,
            module_overrides: dict = ['network', 'key', 'auth', 'namespace', 'serializer'],
            new_event_loop: bool = True,
            save: bool = True


        ) :
        self.protected_modules = module_overrides
        self.module = c.Module()
        input = self.argv()
        args, kwargs = self.parse_args(input)
        
        module_list = c.modules()
        if new_event_loop:
            c.new_event_loop(True)

        fn = None
        module = None
        if len(args) == 0:
            output = c.schema()
        elif len(args)> 0:
            functions = list(set(self.module.functions()  + self.module.get_attributes()))

            args[0] = self.resolve_shortcut(args[0])
            
            # is it a fucntion, assume it is for the module

            module_list = c.modules()

            # handle module/function
            if '/' in args[0]:
                args = args[0].split('/') + args[1:]
                
                
            if args[0] in functions and args[0] not in module_overrides and args[0] not in self.protected_modules:
                # is a function
                module = c.Module
                fn = args.pop(0)
            elif args[0] in module_list:
                # is a module
        
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
            try:
                self.put(f'cli_history/{int(c.time())}', {'input': input, 'output': output})
            except Exception as e:
                pass

        if isinstance(output, type(None)):
            c.print(output)
        else:
            if c.is_generator(output):
                for i in output:
                    c.print(i)
            else:
                c.print(output)
    @classmethod
    def history(cls):
        return cls.ls('cli_history')
    
    @classmethod
    def clear(cls):
        return cls.rm('cli_history')
    



        