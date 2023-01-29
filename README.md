# Commune

This repo intends to standardize all python processes across a peer to peer network. This Allows for developers to seemlessly connect with each other without having to go through an intermediary. 


## Concepts


### Module

A module is a python class that was created to add special functions and abilities to existing python classes in order to make them more interoperable, organized and connectable. The module object is contained in the module.py. 

Converting a python class into a module

```
from mypythonclass import LLMModel
import commune

# Instantiate your python class
model = LLMModel()


# convert it into a module 
model= commune.module(model)

# list functions of this module
model.functions()

# turn it into a remote server (via gprpc):

model.serve()


# list to the remote servers




````


You can serve the model as a grpc server as folows

```

model.serve()

```



### Module Filesystem

The module.py file serves as an anchor in that it organizes future modules in what we call a module filesystem. This is essentially where users will organize their modules based on their location with respect to the module.py file. An example is storing a dataset module in {PWD}/dataset/text, which will have a path of dataset.text . This allows for developers to build whatever ontology or filesystem they want, and the module.py tracks it for you lol.

{image of module filesystem}





## Setup

### From Source

1. clone from github
    ```
    git clone https://github.com/commune-ai/commune.git
    ```
2. Pull the submodules 
    ```
    make pull
    ```

3. Start Docker compose. This builds and deploys all of the containers you need, which involves, commune (main enviornment), ipfs (decentralized storage), subspace (blockchain node), ganache (evm development node for local blockchain)

    ```
    make up
    ```
    - please note that if you are not using a gpu, you will need to comment out the following docker compose code in ./docker-compose.yml

4. Enter the commune evnironment 
    ```
    make enter
    ```



