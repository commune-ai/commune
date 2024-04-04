# Commune Storage Readme

This Python script contains the definition of the `Storage` class, a component utilized for storing and retrieving data. The class definition includes methods concerning the encryption, serialization, and validity verification (staleness) of stored data. The class is a part of the `commune` module.

## Storage Class

The `Storage` class accepts a `Dict` and a `Key` instance for initialization. It provides methods to store and retrieve data while enforcing security measures such as encryption.

1. **`set_storage`** - Updates the initial storage. 
2. **`put`** - Puts value into the storage under a specific key. Allows possibility for encryption.
3. **`state_dict`** - Serializes the current state of the storage object.
4. **`from_state_dict`** - Updates the storage based on a serialized state dictionary.
5. **`save`** - Saves the current state of the storage in JSON format.
6. **`resolve_key`** - Resolves a provided key or defaults to the initialized key.
7. **`get`** - Retrieves data from the storage after verifying the data signature and checking staleness.
8. **`key2address`** - Returns a key to address mapping.
9. **`is_encrypted`** - Checks whether a stored item is encrypted.

## Test and Sandbox

There are also two class methods at the end of the script for testing the `put` and `get` methods of the `Storage` class, and for sandboxing this class.

## Usage 

This Python script is executable. On execution, it will run the `sandbox()` method of `Storage` class. Alternatively, you can import this in your Python program and utilize the `Storage` class for secure storage and retrieval of data.

## Dependencies
This script uses the `commune` and `streamlit` modules. Please ensure these modules are installed before running the script.

Remember, data obtained from `get` method has undergone signature verification and staleness check to ensure security and data integrity.
