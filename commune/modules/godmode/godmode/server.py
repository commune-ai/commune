import commune as c 
class Server:
    """
    A base class that provides fundamental functionality for commune modules.
    """
    def __init__(self, **kwargs):
        """
        Initialize the base class with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Store configuration as a Munch object for dot notation access
        self.model = c.module('openrouter')()
        
        
    def forward(self, text, *args, stream=1,  **kwargs):
        """
        Dynamically call a method of the class.
        Args:
            fn_name (str): Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        Returns:
            Result of the called method
        """
        text = ' '.join([text] + list(args))
        return self.model.forward(text, stream=stream)