# COMMUNE: An Open Python Network

Commune is a protocol that aims to connect all developer tools into one network, fostering a more shareable, reusable, and open economy. It follows an inclusive design philosophy that is based on being maximally unopinionated. This means that developers can leverage Commune as a versatile set of tools alongside their existing projects and have the freedom to incorporate additional tools that they find valuable.

By embracing an unopinionated approach, Commune acknowledges the diverse needs and preferences of developers. It provides a flexible framework that allows developers to integrate specific tools seamlessly while avoiding imposing rigid structures or constraints. This adaptability enables developers to leverage Commune's capabilities in a manner that best aligns with their individual projects and workflows.

The overarching goal of Commune is to create a collaborative ecosystem where developers can easily share, connect, and extend their tools, ultimately fostering innovation and efficiency within the development community. By providing a network that encourages openness and accessibility, Commune empowers developers to leverage the collective knowledge and resources of the community to enhance their own projects.

## Socials

- Twitter: [@communeaidotorg](https://twitter.com/communeaidotorg)
- Discord: [commune.ai](https://discord.gg/DgjvQXvhqf)
- Website: Comming Soon

## Setup

### Setting up Commune with Docker

Install Docker: If you don't have Docker installed on your system, download and install it from the official Docker website: [https://www.docker.com/get-started](https://www.docker.com/get-started).

Clone the Commune Repository: Open your terminal or command prompt and clone the Commune repository from GitHub:

```bash
git clone https://github.com/commune-ai/commune.git
```

```bash
make up
```

To enter the docker container do

```bash
make enter
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

### Setting up Commune Without Docker

Clone the Commune Repository: Open your terminal or command prompt and clone the Commune repository from GitHub:

```bash
git clone https://github.com/commune-ai/commune.git
```

Install Dependencies: Navigate to the cloned Commune repository and install the required dependencies:

```bash
cd commune
pip install -e ./
```

# ENSURE YOU HAVE AN NPM ENVIRONMENT FOR PM2

```bash
chmod +x ./scripts/*
sudo ./scripts/install_npm_env.sh
npm install -g pm2
```

Congratulations! Commune is now set up and running without Docker

## Deploy Your Object From Anywhere

Commune allows developers to deploy, connect, and compose Python objects. The vision of Commune is to create an open ecosystem of Python objects that can serve as APIs for others. Commune provides additional tools through its `Module` object, which seamlessly integrates with any Python class. This means that you do not have to fundamentally change your code when making it public.

To deploy your model as a public server, you can launch it using the following code:

```python
# Give it a name; this will infer the IP and port
MyModel.launch(name='my_model')

# You can also give custom kwargs and args
MyModel.launch(name='my_model::2', kwargs={}, args={})

# Don't like __init__? Start the module from a class method instead
MyModel.launch(name='my_model::2', fn='load_from_name', kwargs={'name': 'model_3'})
```

## Module Namespaces

A module namespace allows you to connect and reference your modules by the name you give them.

## Connecting to a Module

To connect with a module, you can do it as follows. This creates a client that replicates the module as if it were running locally.

```python
my_model = commune.connect('my_model')
# Supports both kwargs and args, though we recommend kwargs for clarity
my_model.forward(input='...')
```

You can also get more information about the module using the `info` function, which is a function from `commune.Module` that wraps over your Python class.

```python
# Get module info
model_info = my_model.info()
```

You can also get the functions and their schemas:

```python
# Get functions (List[str])
my_model.functions()

# Get function schema
my_model.function_schema()
```

### Module Filesystem

The `module.py` file serves as an anchor, organizing future modules in what we call a module filesystem. For example, you can store a dataset module in `{PWD}/dataset/text`, which will have a path of `dataset.text`. The current limitation is to have a config where the name of the config is that of the Python object.

Example:

```bash
model/text/ # model folder (model.text)
    text_model.py # python script for text model
    text_model.yaml # config for module
```

You can get this using the path (`model.text`):

```python
# Get the model class
model_class = commune.module('model.text')

# You can use it locally, obviously
model = model_class()

# Or you can deploy it as a server
model_class.launch(name='model.text')
```

[Insert image of module filesystem]

# Subspace

Subspace is a blockchain that Commune uses for several purposes:

- **DNS for Python**: Decentralized Name Service for deployed objects.
- **Evaluating Performance through Voting**: Stake-weighted voting system for users to evaluate each other instead of self-reported networks. This provides users with

## Register

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

That's it, my friend! You've created a dope module and registered it on the blockchain with the option to customize the name and tag. Now you can share your module with the world and let others benefit from your greatness. Keep on coding and stay fresh!
