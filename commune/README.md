
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

