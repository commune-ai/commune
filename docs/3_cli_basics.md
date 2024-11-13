# The Pythonic CLI

We have a pythonic cli for commune, which is a wrapper around the `c.Module` library. This is a simple way to interact with the commune library. This does not need to be formated like argparse, and is more like a pythonic cli, where you can test out the functions and modules.



```bash
c {module_name}/{function_name} *args **kwargs


```

## Pythoni
You do not need to specify the module when calling the root (name=module) module.
```bash
c {function_name} *args **kwargs
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










The template for the cli is as follows:
```bash
c model.openai/forward text="sup"
```
or 
```bash

c {module_name}/
```

For example, the following command:


```python
import commune as c
c.modules("model")
```

is the same as 

```bash
c modules model
```

## Using the cli to interact with your modules

You can use the cli to interact with your modules. For example, if you have a module called `demo`, you can use the cli to interact with it. 

For instance, to get the config of the model.openai module, you can use the following command:

```bash
c model.openai/config
```

This is the same as the following python code:

```python
import commune as c
c.module("model.openai").config()
```


## Serving 

You can also serve your modules using the cli. For example, if you have a module called `demo`, you can serve it using the following command:

```bash
c demo/serve tag=latest
```

This is the same as the following python code:

```python
import commune as c
c.module("demo").serve(tag="latest")
```




# Misc 


c run-epoch  is c run_epoch # all - is _