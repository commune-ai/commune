# Subspace Module

This involves a runthrough of the main function of subspace. We recommend going through the code if you want to understand the full extent of this module as there are alot of functions

## Register a Module
To register a module, you can use the following command

```
c model.openai register tag=sup api_key=sk-...
```

or

```
c register model.openai tag=sup api_key=sk-...
```

Please make sure you specify a unique tag, as it will not go through if someone else has that name on the subnet. When you deploy the module, the module will be serving locally on your machine and will be accessed by the network.


## Update a Module

To update a module, you can use the following command. At the moment you can update the module's name and address. Please not if you update the name, you will need to restart the server with the new name. This is currently something we want to avoid in the future by having to rename the server without killing it 

```

c update_module model.openai name=model.openai::fam1 address=124.545.545:8080
```



## Check Stats

To check the stats of a module, you can use the following command

```
c stats
```

![Alt text](image.png)

If you want to sync the stats with the network, you can use the following.

```
c stats update=True
```



## Transfer Tokens

To tranfer 100 tokens to a module, you can use the following command

```
c transfer 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S 100 
```


## Stake Tokens

To stake on a module, you can use the following commands:

```
c stake model.openai amount=100
```
or
```
c stake model.openai 100
``` 
or
```
c stake 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S amount=100


## Unstake Tokens
```
To unstake on a module, you can use the following command
```

Unstake 100 tokens from the module with the name model.openai

```
c unstake 100 # unstake 100 tokens from the most staked module
```

Unstake 100 tokens from the module with the name model.openai
```
c unstake 100 model.openai 
```

```
c unstake 100 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S # unstake 100 tokens from the module with the address 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S
```
```



### Unstaking on another Module with your Tokens

To unstake on another module with your tokens, you can use the following command

```
c unstake key=model.openai amount=100 module_key=model.openai.2
```

## Start Local Node

To start a local node, you can use the following command, please ensure you have docker installed.

```
c start_local_node
```

This starts a node with the name alice on the main chain. You can also specify the chain to be main or dev.


## Start a Chain

To start a chain, you can use the following command


This starts a chain with the name alice on the main chain. You can also specify the chain to be main or dev.

```
c start_chain dev valis=4 nonvalis=4
```














