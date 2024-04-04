# Python Code Readme

This Python script is the implementation of the `ObjectServer` class using Ray actors for managing distributed objects across the network nodes.

## Key methods and properties in the `ObjectServer` class include:

- `__init__`: This constructor initializes the ObjectServer instance using a configuration.

- `put`: This method puts an object in the cache dictionary associated with a key and returns its ObjectRef.

- `get`: This method retrieves an item from the cache by its key.

- `get_cache_state`: This method returns the state of the cache.

- `get_fn`: A property decorator that asserts if the passed function is callable and returns it.

- `search`: This method searches for an object in the cache.

- `search_keys`: This method retrieves keys from the cache that satisfy a filter function.

- `pop`: This method removes and returns items from the cache that satisfy a search function.

- `ls`: This method returns a dictionary with the requested key.

- `glob`: This method essentially just calls the `ls` method with the provided key.

- `has`: This method checks if an object with the specified key exists in the dictionary.

## Components

- Ray: It is a flexible, high-performance distributed execution framework. 

- Streamlit: Framework for quickly creating Machine learning and Data Science web applications.

- Commune: A package that simplifies the handling of tasks in distributed systems.

At the end of the script, the `ObjectServerModule` is instantiated and tested. It can be run in Python terminal using the command - `python filename.py`. 

**Please Note**: This code might require knowledge of distributed system and parallel programming in Python using the ray library.