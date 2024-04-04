# Hash Module with Commune Library

This script is a simple demonstration of the Commune's `Module` to build a hash generator class `Hash`. This can be helpful to streamline processes that require a hash function where various hashing options are needed.

## Importing the Libraries
The script starts by importing the Commune library. 

```python
import commune as c
```

## Creating a Hash Class
The main part of the script is the `Hash` class. 

This class carries out hashing function according to the mode specified. It also has some additional class methods, such as to list all the supported hashing modes, testing the hash function, and calling the hash function.

### Hashing Modes 

Through the `hash` method, the class `Hash` support several hashing mode such as 'keccak', 'ss58', 'python', 'md5', 'sha256', 'sha512', 'sha3_512'. 

### hash_modes method
`hash_modes` method is a class method that lists all the supported hash modes.

### Testing
`test` method tests whether the hash method is working for all the hashing modes. This method prints hash result of the tested string with all available hash modes.

### Calling the Hash 

The `__call__` method allows an instance of the `Hash` class to be called like a function, using the defined hash method.


## Main Function
Finally, at the bottom of the script, there's a simple main function to run executed when this script is called independently.

```python
if __name__ == "__main__":
    Hash.run()
```

## Commune Library Dependency

Ensure that the Python environment has the Commune library. It can be installed via pip:

```bash
pip install commune
```

This script provides a simple way to utilize various hash methods from different libraries, testing them and outputting hash for a given input, packaged within a neat and versatile class using the Commune library.