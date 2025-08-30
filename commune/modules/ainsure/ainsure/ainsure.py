import commune as c 
class Base:
    """
    A Ainsure class that provides fundamental functionality for commune modules.
    """
    def __init__(self, **kwargs):
        """
        Initialize the Ainsure class with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Store configuration as a Munch object for dot notation access
        self.model = c.module('openrouter')()
        
        
    def forward(self, module: str='explain', *args, stream=1,  **kwargs):
        """
        Dynamically call a method of the class.
        Args:
            fn_name (str): Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        Returns:
            Result of the called method
        """
        return self.model.forward(f'what does this do? {c.code(module)}', stream=stream)