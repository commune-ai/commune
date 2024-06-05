<div align="center">

# **Commune AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.com/invite/DgjvQXvhqf)
[![Website Uptime](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.communeai.org/)
[![Twitter Follow](https://img.shields.io/twitter/follow/communeaidotorg.svg?style=social&label=Follow)](https://twitter.com/communeaidotorg)

### An Open Modules Network

</div>

Commune is a protocol that aims to connect all developer tools into one network, fostering a more shareable, reusable, and open economy. It follows an inclusive design philosophy that is based on being maximally unopinionated. This means that developers can leverage Commune as a versatile set of tools alongside their existing projects and have the freedom to incorporate additional tools they find valuable.

By embracing an unopinionated approach, Commune acknowledges the diverse needs and preferences of developers. It provides a flexible framework that allows developers to integrate specific tools seamlessly while avoiding imposing rigid structures or constraints. This adaptability enables developers to leverage Commune's capabilities in a manner that best aligns with their individual projects and workflows.

The overarching goal of Commune is to create a collaborative ecosystem where developers can easily share, connect, and extend their tools, ultimately fostering innovation and efficiency within the development community. By providing a network that encourages openness and accessibility, Commune empowers developers to leverage the collective knowledge and resources of the community to enhance their own projects.

# Install

### Setting Up With setup.py

Install setuptools:
If you haven't already installed setuptools, you can do so using pip:

```bash
pip install -r requirements; pip install -e .
```
or 
```bash
./start.sh
```

### Setting Up Commune With Docker

Install Docker: If you don't have Docker installed on your system, download and install it from the official Docker website: [https://www.docker.com/get-started](https://www.docker.com/get-started).

Clone the Commune Repository: Open your terminal or command prompt and clone the Commune repository from GitHub:

```bash
make up 
```

or 
    
```bash
docker-compose build
```

Start Commune: Once the Docker container is built, start Commune by running the following command:

```bash
make start
```
or 
```bash
docker-compose up -d # -d for detached mode
```

To enter the docker container do, and do the following

```bash
make enter
```
or 
```bash
docker exec -it commune bash
```

Then run the following command to sync the network

```bash
c ls
```





To exit the container

```bash
exit
```

Sync Commune with the Network: Inside the Docker container, run the following command to sync Commune with the network:

```bash
c sync
```

Congratulations! Commune is now set up and running inside a Docker container.
Congratulations! Commune is now set up and running without Docker

## Note:

This repo is on the cutting edge of experimentation, so there may be some hiccups along the way. If you're primarily interested in using the core features of the protocol (such as intuitive cli) or seeking a more lightweight implementation, consider installing the [Communex](https://github.com/agicommies/communex) package.

# Key Features

## Module Filesystem

The `module.py` file serves as an anchor, organizing future modules in what we call a module filesystem. The module filesystem is a mix of the commune's core modules with your local modules with respect to your PWD (parent working directory). This allows you to create a module in your local directory and have it be recognized by the commune. For instance if I have a file called `model.py` in my local directory, with the following code:

```python

import commune as c
class Example(c.Module):
    def __init__(self):
        pass

    def predict(self, x):
        return x + 1
```

I can call it from the commune cli by running the following command:

```bash
c model/predict x=1
```
or 
```python
c.call('model/predict', x=1)
```


## Subspace

![Example](https://drive.google.com/uc?export=view&id=1ZqCK-rBKF2p8KFr5DvuFcJaPXdMcISlT)

Subspace is a blockchain that Commune uses for several purposes:

- **DNS for Python**: Decentralized Name Service for deployed objects.
- **Evaluating Performance through Voting**: Stake-weighted voting system for users to evaluate each other instead of self-reported networks. This provides users with

## Register On The Chain

To register a module, do the following

```python
c register {module_path} name={module_name (OPTIONAL)}
```

The module path is specified

Yo, listen up! I'm about to drop some updated knowledge on how to create a dope module and register it on the blockchain. Here's the revised step-by-step guide:

1. **Create Your Module**: Start by creating your own module in Python. It can be anything you want - a model, a service, or some sick functionality. Make sure your module is ready to rock and roll.

2. **Import Commune**: Import the Commune library into your Python code. You'll need it to create and register your module.

```python
import commune as c
```

3. **Define Your Module Class**: Create a class that represents your module. Make sure it inherits from `c.Module`.

```python
class MyDopeModule(c.Module):
    def __init__(self):
        super().__init__()
        # Your module initialization code goes here

    def some_cool_function(self):
        # Your module's cool functionality goes here
        return "I'm bringing the heat!"
```

4. **Register Your Module**: Now it's time to register your module on the blockchain. You have the option to specify a custom name and tag for your module. If you don't provide a custom name, the module will default to the module path. The tag is optional and can be used for versioning or categorization purposes.

To register your module with a custom name and tag, run the following command:

```bash
c register my_module_path name=my_module tag=1
```

Replace `my_module_path` with the actual path to your module file (without the class name), `my_module` with the desired name for your module, and `1` with the desired tag. This will register your module on the blockchain with the specified name and tag.

If you prefer to use the default module path as the name, simply omit the `name` parameter:

```bash
c register my_module_path tag=1
```

# Developement FAQ

- Where can i find futher documentation? This repository folder, [Doc](https://github.com/commune-ai/commune/tree/main/docs).
- Can I install on Windows? Yes, [Guide](https://github.com/OmnipotentLabs/communeaisetup).
- Can I contribute? Absolutely! We are open to all contributions. Please feel free to submit a pull request.


## Testing

We use pytest for testing. To run the tests, simply run the following command:

Make sure you install pytest
```bash
pip install pytest
```
ensure you are in the commune repo

```bash
pytest commune
``` 