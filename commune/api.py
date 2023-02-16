

import commune


class API(commune.Module):
    def __init__(self, module:str = 'module', refresh=False):
        if not commune.actor_exists(module) or refresh:
            commune.launch(name='module', mode='ray')
        self.module = commune.connect(module)
        self.merge(self.module)
        
    @classmethod
    def run(cls): 
        args = cls.argparse()
        self = cls()
        output =getattr(self, args.function)(*args.args, **args.kwargs)  
        commune.log(output, 'green')        


if __name__ == "__main__":
    API.run()