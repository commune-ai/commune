# Commune

Commune is a framework of tools that prevents developers from reinventing the wheel. The design of commune is intended to be maximally unoppinionated, with some minor assumptions about the class. This currently involves wrapping your python class with **commune.Module**. 


```python

import commune

class MyModel(commune.Module):
    def __init__(self, model):
        model = self.model
    def forward(input:torch.Tensor):
        return self.model()

```

## Deploy Your Object From Anywhere

Commune enables developers to deploy, connect and compose python objects. Our vision is to have an open ecosystem of python objects that serve as APIs for others. Commune also provides additional tools through its Module object, which was designed to seemlessly integrate with any python class. This means that you do not have to fundementally change your code when making it public.


To deploy your model as a public server launch it:
```python

# give it a name, this will infer the ip and port
MyModel.launch(name='my_model')

# you can also give custom kwargs, args
MyModel.launch(name='my_model::2', kwargs={}, args={})

# dont like __init__? Start the module from a class method instead
MyModel.launch(name='my_model::2', fn='load_from_name' kwargs={'name': 'model_3'})

```


## Connecting to a Module

To connect with a module, you can do it as follows. This creates a client that replicates the module as if it was running locally. This

```python
my_model = commune.connect('my_model')
# supports both kwargs and args, though we recommend kwargs for clarity
my_model.forward(input = '...') 

```

We want to know more about the module, so lets see it through info, which is a function from commune.Module that wraps over you python class.

```python
# get module info
model_info = my_model.info()
```

We can also get the functions and the functions schema

```python
# get functions (List[str])
my_model.functions()

# get function schema

my_model.function_schema()

```


### Module Filesystem

The **module.py** file serves as an anchor in that it organizes future modules in what we call a module filesystem.  An example is storing a dataset module in {PWD}/dataset/text, which will have a path of dataset.text. The current limitation is to have a config where the name of the config is that of the python object

Example: 
```bash
model/text/ # model folder (model.text)
    text_model.py # python script for text model
    text_model.yaml # config for module
```

Now we can get this using the path (model.text):
```python
    # get the model class
    model_class = commune.module('model.text')

    # you can use it locally obviously
    model = model_class()

    # or you can deploy it as a server
    model_class.launch(name='model.text')


```



{image of module filesystem}



# Subspace

Subspace is our blockchain that is used for several things:
- **DNS for Python**: Decentralized Name Service for deployed object
- **Evaluating Performance through Voting**: Stake weighted voting system for users to evaluate each other instead of self-reported networks. This provides users with an idea of the "best" models. 
- **Subnets**: The ability for users to create their own subnetworks to tackle a specific problem. This is in addition to the default subnet (commune, netuid: 0)
-**IBC for bridging with any Chain (Coming Soon)**
- **Smart Contracts (Coming Soon)**: We intend to support



## Setup
1. clone from github
    ```
    git clone https://github.com/commune-ai/commune.git
    ```
2. install commune
    ```
    make install
    ```
2. update commune
    ```
    commune sync
    ```

### Install as a Docker Container


1. Start docker-compose. This builds and deploys all of the containers you need, which involves, commune (main enviornment), ipfs (decentralized storage), subspace (blockchain node), ganache (evm development node for local blockchain)

    ```
    (sudo) make up
    ```
    - please note that if you are not using a gpu, you will need to comment out the following docker-compose code in ./docker-compose.yml

2. Enter the commune evnironment 
    ```
    (sudo) make enter
    ```

