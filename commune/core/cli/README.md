# The CLI Module

We have a pythonic cli for commune, which is a wrapper around the `c.Module` library. This is a simple way to interact with the commune library. This does not need to be formated like argparse, and is more like a pythonic cli, where you can test out the functions and modules.

There are two paths to your first aergument

c {fn} *args **kwargs  (default module is "module")

or 

c {module}/{fn} *args **kwarrgs

```bash
c {module_name}/{function_name} *args **kwargs
```
```bash
c module/ls ./
```

The root module is specified as the module that is closest to the commune/ repo in this case its commune/module.py

by default the module is isgnored in the naming convention to add ease 

Naming simplifications in commune

commune/module.py --> commune
storage/module.py --> storage
storage/storage/module.py -> storage


if you specifiy a root function in module, then you can call the module directly. 
```bash
c {function_name} *args **kwargs
```

```bash
To get the code of the module

c {module_name}/code
```bash
c module/code
```
or you can call the code function on the root module
```bash

## Pythonic 
You do not need to specify the module when calling the root (name=module) module.
```bash

```
Example 


For example, the following command:
```bash
c ls ./ # 
```
is the same as
```bash
c module/ls ./
```
and
```python
import commune as c
c.ls('./')
```

To make a new module
```
c new_module agi
```
```python
c.new_module("agi")
```


This will create a new module called `agi` in the `modules` directory. 
This will be located in 

to get the config of the model.agi module, you can use the following command:

```bash
c agi/config
```
if you dont have a config or yaml file, the key word arguments will be used as the config.

This is the same as the following python code:
```python

import commune as c
c.module("agi").config()
```


To get the code
```bash
c agi/code
```

```python

import commune as c

class Agi(c.Module):
    def __init__(self, a=1, b=2):
        self.set_config(locals())

    def call(self, x:int = 1, y:int = 2) -> int:
        c.print(self.config)
        c.print(self.config, 'This is the config, it is a Munch object')
        return x + y
    

```

to get the config, which is a yaml, or the key word arguments of the __init__
```bash
c agi/config
```


The 

The commune cli needs to be able to call functions from the modules. This is a simple way to call functions from the modules.
c {modulename}/{fn} *args **kwargs

```bash
c serve module

```
To call the forward function of the model.openai module
```bash
c call module/ask hey # c.call('module/ask', 'hey')
# c.connect('module').ask('hey')
```
If you want to include positional arguments then do it 

```bash

c call module/ask hey stream=1 
# c.call('module/ask', 'hey', stream=1)
# c.connect('module').ask('hey', stream=1)
```


c cmd 


Tricks 

```bash
c # takes you to commune by doing c code ./
```

getting code

c code module/code # gets the code of module/code module
c schema /code gets the code of the module key


cool shortcuts


c module/ cals the module forward function
c module/forward calls the forward cuntion
c module/add a=1 b=1 equals c module/add 1 1 
c # just goes to commune repo



c ai what is the point of love 

Limitatons

- Lists and dictionaries are not supported 
- Only positional arguments are supported
- Only one function can be called at a time


