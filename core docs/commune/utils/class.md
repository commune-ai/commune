# README for Merge Functions

This Python script provides functionalities to merge the attributes and methods of one object into another. It is helpful in cases where you want to compose an object by bringing together functionalities from multiple objects while preserving immutability.

## Code Details

- `merge_dicts(a, b, include_hidden, allow_conflicts)` is a function that merges the attributes of two given Python objects.
- `merge_functions(a, b, include_hidden, allow_conflicts)` is a function that merges the methods of two given Python objects.
- `merge(a, b, include_hidden, allow_conflicts)` is a function that combines the process of merging attributes and methods into one function.

The parameters for each of these functions are:
- `a` and `b` represent the two objects to be merged.
- `include_hidden` is a boolean flag which decides whether to include hidden methods/attributes (those starting with '__').
- `allow_conflicts` is a flag which decides whether to overwrite the existing attributes/methods in the parent object `a`.

## Required Libraries
This script does not require any additional libraries apart from Python's built-in `typing` module.

## Requirements and Installation
Python 3+ is what you need to run this script. No extra installation is required.

## How to Use

You can use the functions in this script by importing them into your project, like so:
```python
from your_module import merge, merge_dicts, merge_functions
```
Subsequently, you can merge two objects (For example 'obj1' and 'obj2') as follows:

```python
merged_obj = merge(obj1, obj2, include_hidden=False, allow_conflicts=False)
```

**Note:** 'your_module' in the import statement should be replaced with the name of your Python file. Always remember to test your code thoroughly before deploying it to any production environments.
