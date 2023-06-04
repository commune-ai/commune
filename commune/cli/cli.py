
import commune as c

class CLI(c.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    # 
    default_fn = 'help'
    def __init__(
            self,
            config: c.Config = None,

        ) :
        c.new_event_loop(True)
        self.module = c.Module()
        args, kwargs = self.parse_args()

        

        module_list = c.module_list()

        fn = None
        module = None
        if len(args) == 0:
            result = c.schema()
        elif len(args)> 0:
            
            

            module_list = self.module_list()
            functions = list(set(self.module.functions()  + self.module.get_attributes()))
        

            args[0] = self.resolve_shortcut(args[0])
            
            if args[0] in functions:
                module = c.Module
                fn = args.pop(0)
            else:
            
                module_list = c.module_list()
                
                if args[0] in module_list:

                    module = args.pop(0)
                    module = c.module(module)
                else:
                    
                        
                    namespace = self.namespace(update=False)
                    if args[0] in namespace:
                        module = args.pop(0)
                        module = c.connect(module)

                    else: 
                        raise Exception(f'No module, function or server found for {args[0]}')
            
            if fn == None:
                if len(args) == 0:
                    fn = self.default_fn 
                else: 
                    fn = args.pop(0)
                    
                    
                    
            result = getattr(module, fn)
            if callable(result):
                result = result(*args, **kwargs)
                
        else:
            raise Exception ('No module, function or server found for {args[0]}')


        if not isinstance(result, type(None)):
            self.print(result)
    

    
    shortcuts = {
        'bt': 'bittensor',
        'hf': 'huggingface'
    }
    @classmethod
    def resolve_shortcut(cls, name):
        if name in cls.shortcuts:
            return cls.shortcuts[name]
        else:
            return name
        
