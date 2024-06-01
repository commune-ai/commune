# Staking 
## Stake Tokens

To stake on a module, you can use the following commands:

```
c.stake(module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S, amount=100,  key='module')
```

## Unstake Tokens

The following unstakes 100 tokens from the module with the address 5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S
The default key is the root key (module) and the default netuid is 0 (commune).

```python
c.unstake(module=5HYnok8FmBEx9AekVUPzff9kW7ymyp9mrucQTeqTLfJoHu1S , amount=10, netuid=0,  key='module')
```

Unstake 100 tokens from the module with the name model.openai
This uses the name2key function to get the key from the name. 

NOTE:
Please note that is is always more secure to use the address as the key, as the name can be changed by the user, and you can accidentally stake the incorrect key if your name2key is not up to date. 

```python
c.unstake(module=model.openai, amount=100, netuid=0, key='module')
```

### Stake Many

To stake multiple modules at once do the following

```bash 
c.stake_many(modules=[model1,model2].amounts=[10,20], netuid=10)
```

or if you want to specify the same amount, just do the amounts as an integer

```bash 
c.stake_many(modules=[model1,model2], amounts=[10, 10], netuid=0)
```


### Unstake Many

To stake multiple modules at once do the following

```bash 
c.unstake_many(modules=[model1,model2], amounts=[10,20])
```

or if you want to specify the same amount, just do the amounts as an integer

```bash 
c.stake_many(modules=[model1,model2], amounts=10)
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
c stake_transfer vali.text.truthqa::commie1 vali::project_management amount=0.01 # default netuid=0
```

