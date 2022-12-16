<div align="center">

# **WholeTensor** <!-- omit in toc -->

</div>

***
## Summary 

This is a combination of bittensor and subtensor for those wanting to run a local subtensor #MONOREPO


The main feature of this repo is the dashboard that can be used to better understand the bittensor network.

***

## Setup

1. Clone Repo and its Submodules

```
git clone https://github.com/commune-ai/wholetensor.git
cd wholetensor
git submodule update --init --recursive
```

2. Spinnup Docker Compose
```
make up
```

3. Run the Streamlit app
```
make app
```


## Commands

- Run 
    
     ```make up```
-  Enter Backend 
    
     ``` make bash arg=backend```
-  Enter Subtensor 
    
     ``` make bash arg=subtensor```


- Run Streamlit Server
    
     ``` make app ```



## Dashbaord Info
**MODULE_PATH = './backend/commune/bittensor/module.py'**



- Select your Network and Block to View The     Dashboard


- Filter Nodes to reduce computation
    - random sample each node
    - filter based on metric (ie rank, trust, consensus)

- Custom Plots
    - the dashbaord allows for custom plots done primarely in streamlit


Improvements Pending:
- Avoid rerendering and computation with using a background process that accepts api requests from streamlit
- allow developers add their own plots via a dashoard
- allow developers to save their plots as json's which can then be stored in ipfs
- allow developers to run their own ipfs nodes




**Modules**
- ClientManager
    - manages a connection to various clients that can help store and retreive data from different sources/databases/apis/etc

- ConfigLoader
    - A module that is dedicated to loading config files with some additional funcitonality that allows for composing and referencing multiple configs and variables within configs

- Module
    - The base process that connects to configloader, clients, and supports many-to-many functions for composable DAGs. This is used as the parent process for BittensorModule








