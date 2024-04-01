# Pickle Utility Functions

This module provides utility functions for loading and saving data with pickle, a popular serialization format in Python. The functions ensure error-free data handling with optional verbosity for tracking file paths.

## Functions:

### `load_pickle(file_path: str, verbose: bool) -> object`

This function loads a pickle file from the specified file path and returns the deserialized object. If `verbose` is `True` (by default), it prints the file path of the loaded file.

### `dump_pickle(object: object, file_path: str, verbose: bool)`

This function saves an object to a pickle file at the specified file path. Before dumping the object, it ensures that the directory for the file path exists. If `verbose` is `True` (by default), it prints the file path of the saved file.

## Usage:

These functions are useful for saving intermediate results or data transformations in a persistent manner. By pickling data, you can store complex objects like trained machine learning models or large processed datasets and reload them later as needed.
