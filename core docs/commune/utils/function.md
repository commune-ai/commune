# Python Utility Functions for Class and Function Manipulation

This Python module provides a collection of utility functions intended to manipulate class objects and their associated functions. The utilities help with inspecting, retrieving, and filtering properties of classes, functions, and modules.

## Main Functions:

### `get_parents(obj: Any) -> List[str]` 
Returns the list of parent classes for a given class object.

### `get_function_schema(fn=None, include_self=True, defaults_dict=None,*args, **kwargs)`
Returns a dictionary representing the schema of a given function, including information about input and output types.

### `get_module_function_schema(module, completed_only=False)`
Retrieves the schema (input and output types) of all callable functions within a given module.

### `get_self_methods(cls: Union[str, type])`
Retrieves methods that have 'self' as a parameter from a given class.
      
### `get_function_signature(fn) -> dict`
Returns a dictionary of function signature.

### `get_class_methods(cls: Union[str, type]) -> List[str]`
Returns class methods from a given class.

### `get_function_input_variables(fn) -> dict`
Returns function's input variables.

### `is_full_function(fn_schema)`
Checks if a function schema is 'full', i.e., if it has both input and an output with non-null types. 

### `try_fn_n_times(fn, kwargs:Dict, try_count_limit: int = 10)`
Tries a function certain number of times specified by 'try_count_limit' before raising an exception.

## Note:

These utilities are mainly intended for advanced usage of Python's class and function capabilities, and are applicable to a wide range of environments where complex class and function manipulation or introspection is required.