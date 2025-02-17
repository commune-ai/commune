
import commune as c

class Base:
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
        self.config = c.munch(kwargs)
        
    def call(self, fn_name: str, *args, **kwargs):
        """
        Dynamically call a method of the class.
        Args:
            fn_name (str): Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        Returns:
            Result of the called method
        """
        if hasattr(self, fn_name):
            return getattr(self, fn_name)(*args, **kwargs)
        raise AttributeError(f"Method {fn_name} not found")

    def get_config(self) -> dict:
        """
        Get the current configuration.
        Returns:
            dict: Current configuration
        """
        return self.config

    def update_config(self, **kwargs):
        """
        Update the configuration with new values.
        Args:
            **kwargs: New configuration parameters
        """
        self.config.update(kwargs)

    def test(self) -> bool:
        """
        Basic test method to verify the class is working.
        Returns:
            bool: True if test passes
        """
        try:
            # Basic functionality test
            self.update_config(test_key="test_value")
            assert self.get_config().test_key == "test_value"
            return True
        except Exception as e:
            c.print(f"Test failed: {str(e)}", color='red')
            return False

    @classmethod
    def help(cls):
        """
        Display help information about the class.
        """
        c.print(cls.__doc__, color='green')
        c.print("\nAvailable methods:", color='yellow')
        for method_name in dir(cls):
            if not method_name.startswith('_'):
                method = getattr(cls, method_name)
                if callable(method):
                    c.print(f"- {method_name}: {method.__doc__}", color='cyan')
