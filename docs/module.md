# Module Basics

In this tutorial, we'll explore how to use the `commune` library for module management in Python. The `commune` library provides functionalities for managing and serving code modules easily.

## Table of Contents
- [Finding Your Module](#finding-your-module)
- [Module Management](#module-management)
- [Serving](#serving)

---

## Finding Your Module

You can use the following steps to find and work with modules using the `commune` library.

## New Module Creation
To create a new module, you can use the `commune` command line tool:

```bash
c new_module agi
```

### Searching for a Specific Module
To search for a specific module, you can use the `c.modules()` function with a search query:

```python
c.modules('model.openai')
```
OUTPUT
```python
['model.openai']
```

### Viewing Module Info
You can view the information of a specific module using the `info()` method:

```python
model_openai = c.module('model.openai')
c.print(model_openai.info(*args, **kwargs))
```
or 
```bash
c.print(c.call('model.openai/info', *args, **kwargs))
```


---

## Module Management

Once you've found your module, you can manage it using the following steps.

### Accessing a Module
You can access a module using the `c.module()` function:

```python
demo = c.module('demo')
c.print('## Code for demo module')
c.print(demo.code())
```

### Viewing Module Config
You can view the configuration of a module using the `config()` method:

```python
demo.config()
```

OUTPUT
```python
{
    'name': 'demo',
    'version': '0.1.0',
    'description': 'A demo module for testing purposes',
    'author': 'John Doe',
    'email': '
    'license': 'MIT',

}
```

This is the yaml file if the module has a config file stored in the same directory as the module, otherwise it will be the key word arguments of the __init__ method of the module.


### Listing Module Functions
To list the functions of a module, use the `fns()` method:

```python
demo_functions = demo.fns()
c.print(demo_functions)
```

### Searching for a Function
To search for a specific function within a module, use the `fns()` method with a search query:

```python
function_search_query = 'bro'
matching_functions = demo.fns(function_search_query)
c.print(matching_functions)
```

### Getting Function Schema
You can retrieve the schema of a specific function using the `schema()` method:

```python
function_name = 'bro'
function_schema = demo.schema(function_name)
c.print(function_schema)
```
---

This concludes our tutorial on module management using the `commune` library. You've learned how to find modules, manage their functions, serve them, and interact with served modules. This library can greatly simplify the process of managing and deploying code modules in your projects.
```

Feel free to use and adapt this markdown document for your tutorial needs. Make sure to adjust any details as necessary and include code snippets or explanations for each step to ensure clarity and comprehensiveness.