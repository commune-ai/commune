# Custom class for Commune Module

This is a sample Python code for a custom class `Custom` that represents a module in the commune Python package. This class allows you to dynamically add Python functions to your commune module, check if a function exists, whitelist or blacklist functions, and delete functions.

## Importing the module

```python
import commune as c
```

The commune module must be imported for this custom class to work. The custom class itself is a subclass of the `c.Module` class from the commune package.

## Creating an instance

```python
custom = Custom()
```

You instantiate the class like so. The constructor accepts any number of keyword arguments, and stores them in the 'config' attribute for this instance.

## Adding a function

```python
def my_function():
    return "Hello, World"

custom.add_fn(my_function)
```

The `add_fn` function accepts a function object as an argument, and adds it as an attribute of the custom module instance using the function name as the attribute name.

## Checking if a function exists

```python
is_exists = custom.fn_exists('my_function')
```

The `fn_exists` function checks if a function with the given name exists in the custom module.

## Whitelisting a function

```python
whitelist_response = custom.whitelist_fn('my_function')
```

The `whitelist_fn` function adds the function name to the whitelist.

## Blacklisting a function

```python
blacklist_response = custom.blacklist_fn('my_function')
```

The `blacklist_fn` function adds the function name to the blacklist.

## Removing a function

```python
removal_response = custom.rm_fn('my_function')
```

The `rm_fn` function removes a function from the module.

## Adding multiple functions

```python
def function1():
    pass

def function2():
    pass

custom.add_fns(function1, function2)
```

The `add_fns` function accepts one or more function objects as arguments, and adds them to the custom module.

## Adding a module

```python
module = c.module('module_name')()
custom.add_module(module)
```

The `add_module` function accepts a module object from the commune package, and adds all its functions to the custom module.

## Adding several modules

```python
module1 = c.module('module1')()
module2 = c.module('module2')()

custom.add_modules(module1, module2)
```

The `add_modules` function accepts one or more module objects as arguments and adds all their functions to the custom module.

## Testing the code

```python
assert custom.test()
```

The `test` function is a sample test case. It creates a custom module, adds two functions, and asserts that they work as expected.
