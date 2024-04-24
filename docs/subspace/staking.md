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

c update_module module=model.openai name=model.openai::fam1 address=124.545.545:8080 delegation_fee=10
```



## Stake Tokens

To stake on a module, you can use the following commands:

```
c stake 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S 100
```

## Unstake Tokens

The following unstakes 100 tokens from the module with the address 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S

```
c unstake 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S 10
```


Unstake 100 tokens from the module with the name model.openai
```
c unstake module=model.openai amount=100
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


### Unstake Many

To stake multiple modules at once do the following

```bash 
c stake_many modules=[model1,model2] amounts=[10,20]
```

or if you want to specify the same amount, just do the amounts as an integer

```bash 
c stake_many modules modules=[model1,model2] amounts=10
```


## List Your Staked Modules 


c staked search=vali netuid=1

    dividends                    stake_from    delegation_fee          stake    vote_staleness
11  vali.text.truthqa::commie1   0.016297      836737               5  1543452              49
50    vali::project_management   0.014511      821491             100   835445              47
47            vali::stakecomai   0.014252      793614              20   814443              85
32               vali::comchat   0.018128      614167               5   963592               3
58  vali.text.realfake::commie   0.015503      608276               5   932424              49


Transfer stake from one module to another, if you dont specify the amount, the default is the entire stake towards the module

```bash
c transfer_stake vali.text.truthqa::commie1 vali::project_management amount=0.01 # default netuid=0
```

