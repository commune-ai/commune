

import commune


class API(commune.Module):
    def __init__(self, module:str = 'module'):
        if not commune.server_exists(module):
            commune.launch(name='module')
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