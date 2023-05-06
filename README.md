Commune is a framework that provides developers with a set of tools to prevent them from reinventing the wheel. The design of Commune is maximally unopinionated, with some minor assumptions about the class. To use Commune, you need to wrap your Python class with `commune.Module`.

## Setup

To use Commune, you need to follow these steps:

1. Clone the Commune repository from Github:
```
git clone https://github.com/commune-ai/commune.git
```
2. Install Commune:
```
make install
```
3. Sync Commune with the network:
```
commune sync
```

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

{image of module filesystem}

# Subspace

Subspace is a blockchain that Commune uses for several purposes:

- **DNS for Python**: Decentralized Name Service for deployed objects.
- **Evaluating Performance through Voting**: Stake-weighted voting system for users to evaluate each other instead of self-reported networks. This provides users with an idea of the "best" models.
- **Subnets**: The ability for