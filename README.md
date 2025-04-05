
# What is a module

In this tutorial, we'll explore how to use the `commune` library for module management in Python. The `commune` library provides functionalities for managing and serving code modules easily.

To know what a module is make sure you have commune installed.

To start a new module

c new_module agi

This will create a new module called agi in the modules directory of the commune project.


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

```python
c.new_module('agi')
```

```bash
{
    'success': True,
    'path': '/Users/salvivona/commune/agi',
    'module': 'agi',
    'class_name': 'Agi',
    'msg': ' created a new repo called agi'
}
```
c serve agi

{
    'success': True,
    'name': 'agi',
    'address': '0.0.0.0:50129',
    'kwargs': {}
}

c namespace


### Searching for a Specific Module
To search for a specific module, you can use the `c.modules()` function with a search query:

```bash

c modules model.openai
```

```python
```

```python
c.modules('model.openai')
```
OUTPUT
```python
['model.openai']
```


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
c.new("agi")
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





Limitatons

- Lists and dictionaries are not supported 
- Only positional arguments are supported
- Only one function can be called at a time





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
demo.fns()
```
['test', 'forward']

### Searching for a Function
To search for a specific function within a module, use the `fns()` method with a search query:

```python

matching_functions = demo.fns('forw')
c.print(matching_functions)
```
['forward']

### Function Schema
You can retrieve the schema of a specific function using the `schema()` method:

```python
c.module('model.openai').schema()
```


{
    '__init__': {'input': {'a': 'int'}, 'default': {'a': 1}, 'output': {}, 'docs': None, 'type': 'self'},
    'call': {'input': {'b': 'int'}, 'default': {'b': 1}, 'output': {}, 'docs': None, 'type': 'self'}
}

---

This concludes our tutorial on module management using the `commune` library. You've learned how to find modules, manage their functions, serve them, and interact with served modules. This library can greatly simplify the process of managing and deploying code modules in your projects.
```

Feel free to use and adapt this markdown document for your tutorial needs. Make sure to adjust any details as necessary and include code snippets or explanations for each step to ensure clarity and comprehensiveness.

## Install

**Environment Requirements**
python 3.10>=
nodejs 14>=
npm 6>=

1. **Clone the Commune Repository**: Open your terminal or command prompt and clone the Commune repository from GitHub:

```bash
git clone https://github.com/commune-ai/commune.git
```

```bash
pip install -e ./commune
```

To make sure it is working Run the tests, and they should all pass
```bash
c test # runs pytest commune/tests
```
## Running a Docker Container with Commune

Ensure that you have Docker installed on your machine. If you don't, you can follow the official Docker installation guide for your operating system.


3. **Build the Docker Image**: Navigate to the cloned Commune repository and build the Docker image using the provided `Dockerfile`, This can be done via the docker-compsoe file.:

# you may have to 

```
make start
```

To ensure the container is running, you can run the following command:

```bash
make test
```

4. **Start Container**: Start a Docker container with the Commune image:

```
make start
```

5. **Enter the Container**: Enter the Docker container:

```bash
make enter # or docker exec -it commune bash
```
To exit the container, run the following command:
```bash
exit
```

To run commands inside the container, you can use the following command:

```bash
docker exec -it commune bash -c "c modules"
```

To Kill the container, run the following command:
```bash
make down
```

Congratulations! Commune is now set up and running inside a Docker container.

## Setting up Commune Without Docker

2. **Install Dependencies**: Navigate to the cloned Commune repository and install the required dependencies:

```
cd commune
pip install -e ./
```

3. **install npm pm2**
This is required for the webserver to run
```bash 
chmod +x ./run/* # make sure the scripts are executable
sudo ./run/install_npm_env.sh # install npm and pm2 (sudo may not be required)
```

4. Check if commune is installed correctly, try running the following command
```bash
c modules
```

That's it! Commune is now set up and ready to roll on your local machine.

Whether you choose to set up Commune with Docker or without it, you're all set to leverage the power of Commune and connect your Python objects in a collaborative ecosystem. Have fun exploring the possibilities and making the most out of Commune!

Note: Make sure to follow the official Commune documentation for detailed instructions and additional configuration options.


## Port Range:

```bash

# check the port range
c port_range
```

```bash
c set_port_range 8000 9000 # set the port range to 8000-9000
```



The modules folder

The modules folder is default in ~/modules and contains the code from any module you import 
it is by default stored in ~/modules which can differ depending on your system (inside docker it is /root) while on a macbook its /home/{username}, you can define this by c.home_path




# Scripts




## Scripts Overview

### build.sh
Builds a Docker image for the project.
```bash
./run/build.sh [name]  # name is optional, defaults to repository name
```

### enter.sh
Enters a running Docker container in interactive mode.
```bash
./run/enter.sh   # name is optional, defaults to repository name
```

### install.sh
Sets up the development environment by installing required dependencies:
- npm
- pm2
- Python3
- pip3
- Installs the project as a Python package

```bash
./run/install.sh
```

### start.sh
Starts a Docker container with the following features:
- Host network mode
- Auto-restart unless stopped
- Privileged mode
- 4GB shared memory
- Mounted volumes for app and configuration
- Docker socket access

```bash
./run/start.sh [name]  # name is optional, defaults to repository name
```

### stop.sh
Stops and removes a running Docker container.
```bash
./run/stop.sh   # name is optional, defaults to repository name
```

### test.sh
Runs project tests in a temporary Docker container.
```bash
./run/test.sh
```

## Features

- Automatic repository name detection
- Cross-platform support (Linux, MacOS, Windows)
- Docker container management
- Development environment setup
- Test automation

## Requirements

- Docker
- bash shell
- Package managers (apt, brew, or choco depending on OS)

## Usage

1. Clone the repository
2. Run `./run/install.sh` to set up the development environment
3. Use other scripts as needed for building, starting, stopping, or testing

## Notes

- All scripts use the repository name as the default container/image name
- Custom names can be provided as arguments to most scripts
- The project is automatically installed as a Python package during setup