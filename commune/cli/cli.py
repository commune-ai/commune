

import argparse
import commune
from typing import List, Optional
import json


class CLI(commune.Module):
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __init__(
            self,
            config: commune.Config = None,

        ) :
        commune.new_event_loop(True)
        self.module = commune.Module()
        args, kwargs = self.parse_args()

        

        module_list = commune.module_list()

        fn = None
        module = None
        if len(args)> 0:
    
            namespace = self.namespace(update=False)
            
 

            module_list = self.module_list()
            functions = list(set(self.module.functions()  + self.module.get_attributes()))
            module_options = (list(namespace.values()) + list(namespace.keys()))
            args[0] = self.resolve_shortcut(args[0])
            
            candidates = dict(
                            functions=[f for f in functions if f == args[0]] ,
                            modules=[m for m in module_list if m == args[0]],
                            servers=[s for s in module_options if args[0] == s],
            )
            
            if len(candidates['functions'])>0:
                module = self.module
                fn = fn if fn != None else  args.pop(0)
                fn = candidates['functions'][0]
            elif len(candidates['modules'])>0:
                if module == None:
                    module = args.pop(0)
                else:
                    args.pop(0)
                fn = fn if fn != None else args.pop(0)
                module = commune.module(module)
            elif len(candidates['servers'])>0:
                if module == None:
                    module = args.pop(0)
                else:
                    args.pop(0)
                fn = fn if fn != None else args.pop(0)
                module = commune.connect(module)

            else: 
                raise Exception(f'No module, function or server found for {args[0]}')
        
        result = getattr(module, fn)
        if callable(result):
            result = result(*args, **kwargs)


        self.print(result)
    
    def catch_ip(self):
        result = None
        if len(args[0].split(':')) == 2:
            # commune module:fn args kwargs 
            module_fn = args.pop(0)
            module, fn = module_fn.split(':')
            module = commune.connect(module)
            result = module.remote_call(fn, *args,**kwargs)
            
            
        return result
    
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
        
