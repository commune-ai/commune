<div align="center">

# **Commune AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Discord Chat](https://img.shields.io/badge/discord-join%20chat-blue.svg)](https://discord.com/invite/DgjvQXvhqf)
[![Website Uptime](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)](https://www.communeai.org/)
[![Twitter Follow](https://img.shields.io/twitter/follow/communeaidotorg.svg?style=social&label=Follow)](https://twitter.com/communeaidotorg)

### An Open Modules Network

</div>

Commune is a protocol that aims to unify all developer tools into a singular network, fostering a more collaborative, reusable, and open economy. By adopting a non-biased design approach, Commune can be used as a versatile toolset in synchrony with current projects, also providing the liberty to blend and utilize other tools as needed.

Commune respects the various needs and preferences of developers, offering a flexible framework for integrating specific tools smoothly without imposing rigid structures or constraints. This level of adaptability enables developers to harness Commune's capabilities aligning perfectly with their individual projects and workflows.

The foremost objective of Commune is to foster a collaborative ecosystem where developers can quickly share, connect, and enhance their tools, promoting innovation and efficiency within the development community. By providing a network that underpins transparency and accessibility, Commune encourages developers to leverage the shared knowledge and resources of the community to improve their projects.

# Installation

### Setting Up With setup.py

Firstly, install setuptools using pip:

```bash
pip install -r requirements; pip install -e .
```
or 
```bash
./start.sh
```

### Setting Up Commune With Docker

Firstly, Install Docker from the official Docker website: [https://www.docker.com/get-started](https://www.docker.com/get-started). Then, clone the Commune repository from GitHub:

```bash
make up 
```
or 
    
```bash
docker-compose build
```

To start Commune, run the following command:

```bash
make start
```
or 
```bash
docker-compose up -d # -d for detached mode
```

To connect to the docker container, run the following command:

```bash
make enter
```
or 
```bash
docker exec -it commune bash
```

Next, run the following command to sync the network:

```bash
c ls
```

To disconnect from the container:

```bash
exit
```

In the Docker container, run the following command to sync Commune with the network:

```bash
c sync
```

Congratulations! Commune is now set up and running either with Docker or standalone.

## Note:

This repo is on the cutting edge of experimentation, so you might encounter some bugs along the way. If you're primarily interested in using the core features of the protocol (such as user-friendly cli) or a minimalist implementation, consider installing the [Communex](https://github.com/agicommies/communex) package.

# Features

## Module Filesystem

The `module.py` file serves as a root directory, organizing future modules in what we call a module filesystem. For example, you can store a dataset module in `{PWD}/dataset/text`, which will have a path of `dataset.text`. The current limitation is to have a configuration where the name of the config is that of the Python object.

## Subspace

![Example](https://drive.google.com/uc?export=view&id=1ZqCK-rBKF2p8KFr5DvuFcJaPXdMcISlT)

Subspace is a blockchain that Commune uses for multiple reasons:

- **Python DNS**: Decentralized Name Service for deployed objects.
- **Performance Evaluation through Voting**: Stake-weighted voting allows users to evaluate each other in place of self-reported networks. This provides users with precise measures of performance.

## Register On The Chain

To register a module, execute the following command:

```python
c register {module_path} name={module_name (OPTIONAL)}
```

The module path is defined by:

1. **Create Your Module**: Develop your own module (a model, a service, or functionality) in Python. Ensure the module is functional and ready for deployment.
2. **Import Commune**: Import the Commune library into your Python script, as it's necessary for creating and registering the module:

```python
import commune as c
```
3. **Define Your Module Class**: Next, create a class representing the module. This class should inherit from `c.Module`:

```python
class MyDopeModule(c.Module):
    def __init__(self):
        super().__init__()
        # Initialize your module code here

    def some_cool_function(self):
        # Define your module's functionality here
        return "I'm bringing the heat!"
```

4. **Register Your Module**: Register your module on the blockchain. You can specify a custom name and tag for your module. If not specified, the module name will default to the module path. Moreover, the tag is optional and can be used for versioning or categorization of the module.

To register your module with a custom name and tag, run the following command:

```bash
c register my_module_path name=my_module tag=1
```

Ensure to replace `my_module_path` with the actual module file path (except the class name), `my_module` with the desired module name, and `1` with the desired tag. This command will register your module on the blockchain under the specified name and tag.

If you prefer to use the default module path as the name, simply exclude the `name` parameter:

```bash
c register my_module_path tag=1
```

# Development FAQ

- **Where can I find further documentation?** Go to this repository folder, [Doc](https://github.com/commune-ai/commune/tree/main/docs).
- **Can I install on Windows?** Yes. Use this [Guide](https://github.com/OmnipotentLabs/communeaisetup).
- **Can I contribute?** Absolutely! We are open to all contributions. Please feel free to submit a pull request.
