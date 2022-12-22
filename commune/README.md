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







