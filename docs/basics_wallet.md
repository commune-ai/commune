# Subspace Module

This involves a runthrough of the main function of subspace. We recommend going through the code if you want to understand the full extent of this module as there are alot of functions


## ROOT KEY

The root key is what you should be using for managing your commune modules. by default this is the following.

```bash 
c root_key
```

```bash
<Keypair (address=5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC, path=module,  crypto_type: SR25519)>
```


## getting stats on the network

You can get the stake 

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


## Syncing with the network 

To ensure you are synced with the blockchain you must run the sync loop for subspace.
This loop periodically syncs with the network in the bacground every interval=60 seconds

```bash
c s loop interval=60
```
Result
```bash
{'success': True, 'msg': 'Launched subspace::loop', 'timestamp': 1702854431}

```




## Check Stats

To check the stats of a module, you can use the following command

```
c stats
```

![Alt text](image.png)

If you want to sync the stats with the network, you can use the following.

```
c stats
```



## Transfer Tokens

To tranfer 100 tokens to a module, you can use the following command

```
c transfer 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S 100 
```


## Stake Tokens

To stake on a module, you can use the following commands:

```
c stake module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S amount=100
```
You can also use the name of th emodule
```
c stake module=model.openai amount=100
```

or if you want to use the full amount then you can just leave the amount as blank 
```
c stake module=model.openai
```

### Stake Many

To stake multiple modules at once do the following

```bash 
c stake_many modules=[model1,model2] amounts=[10,20]
```

or if you want to specify the same amount, just do the amounts as an integer

```bash 
c stake_many modules modules=[model1,model2] amounts=10
```





## Unstake Tokens

The following unstakes 100 tokens from the module with the address 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S

```
c unstake  module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S amount=10
```


Unstake 100 tokens from the module with the name model.openai
```
c unstake module=model.openai amount=100
```

### Unstake Many

To stake multiple modules at once do the following

```bash 
c stake_many modules=[model1,model2] amounts=[10,20]
```

or if you want to specify the same amount, just do the amounts as an integer

```bash 
c stake_many modules modules=[model1,model2] amounts=10
```
