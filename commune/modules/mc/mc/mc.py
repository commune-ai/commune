import commune as c 
class Base:
    """
    A base class that provides fundamental functionality for commune modules.
    """
    def __init__(self, key=None, **kwargs):
        """
        Initialize the base class with configurable parameters.
        Args:
            **kwargs: Arbitrary keyword arguments to configure the instance
        """
        # Store configuration as a Munch object for dot notation access
        self.model = c.mod('openrouter')()
        self.store = c.mod('store')('~/.mc2')
        self.key = c.key(key)
        
        
    def reg(self,  key, value):
        """
        Register a key-value pair in the model's configuration.
        Args:
            key (str): The key to register.
            value: The value associated with the key.
        """
        
        self.model.config[key] = value
    def get(self, key, default=None):
        """
        Retrieve a value from the model's configuration by key.
        Args:
            key (str): The key to look up.
            default: The default value to return if the key is not found.
        """
        return self.store.get(key, default)
    
    def put(self, key, value):
        """
        Save a key-value pair to the model's configuration.
        Args:
            key (str): The key to save.
            value: The value associated with the key.
        """
        self.verify_key(key)
        self.store[key] = value
        self.store.put(key, value)
        return self.store[key]

    def verify_key(self, key):
        """
        assert its sr25519
        """
        return c.valid_ss58_address(key)

        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("Key must be a 64-character hexadecimal string.")
        try:
            int(key, 16)
        except ValueError:
            raise ValueError("Key must be a valid hexadecimal string.")

    