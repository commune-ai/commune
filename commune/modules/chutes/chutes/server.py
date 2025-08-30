import commune as c 
class Server:
    """
    A base class that provides fundamental functionality for commune modules.
    """
    my_module_name = __file__.split('/')[-2]
    url = 'chutes.ai'
    def __init__(self, **kwargs):
        """
        Initialize the base class with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Store configuration as a Munch object for dot notation access
        self.model = c.module('openrouter')()
        
        
    def forward(self, module: str= None, *args, stream=1,  **kwargs):
        """
        explains the module and its functionality.
        """
        module = module or self.my_module_name
        return self.model.forward(f'what does this do? {c.code(module or my_module_name)}', stream=stream)

    