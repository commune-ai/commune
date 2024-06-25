# MODULE ROUTES YAML

# this maps the routes to the functions in the module so that anyone can call them from the module base class 
# or from the command line.
# for instance, instead of calling `c subspace:query` you can call `c query` from the module base class or from the
# Entry types
# 1. str: a string that is the name of a function in the module
# 2. dict(from: str, to: str): a dictionary that maps a function name to another function name in the module
# - 1 to 1 mapping: {from: 'foo', to: 'bar'} maps the function 'foo' to 'bar'
# - 1 to many mapping: {from: 'foo', to: ['bar', 'baz']} maps the function 'foo' to 'bar' and 'baz'
# 3. list: a list of function names in the module (length == 2 only)
# - 1 to 1 mapping: ['foo', 'bar'] maps the function 'foo' to 'bar'
# - 1 to many mapping: ['foo', ['bar', 'baz']] maps the function 'foo' to 'bar' and 'baz'