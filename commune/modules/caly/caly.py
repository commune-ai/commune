import commune as c 
class Caly:
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
        
        
    def forward(self, *extra_text, **kwargs):
        """
        Forward the goals to the model.
        Args:
            *goals: Goals to be forwarded
            **kwargs: Additional keyword arguments for the model
        Returns:
            The result of the model's forward method
        """
        return self.model.forward(*goals, **kwargs)