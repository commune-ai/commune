# Commune

Commune is an open-source framework for creating modular, reusable, and interoperable machine learning modules. It provides a way for developers to wrap any machine learning tool into a module, organize them into a module file system (their own module hub), and expose them as public endpoints with access control. Modules are represented as a folder with a main python script and a configuration file, making them portable across machines. The framework includes a module manager API for managing running modules, connecting modules locally or over the wire, and a queue server for managing communication between modules. Modules can interact with smart contracts and be stored on decentralized file storage systems for monetization. Modulus aims to provide full autonomy and intellectual property for developers and avoid platform lock-ins.



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
model.get_functions()

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


## Docker Compose Environment Details

Available Containers In Compose


## Make Function Helpers

Start docker compose
```
make up
```

Stop docker compose

```
make down
```


Enter the Commune Docker Env
```
make enter
```

Get the logs of the container
```
docker logs {}
```





