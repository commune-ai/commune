

## Deploying a Validator Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.

### Step 1: Registering a Validator

To register a validator with a specific tag, use the following CLI command:

```bash
c vali register tag=whadup subnet=commune
or 
c serve vali::whadup netuid=0 subnet=commune
c register vali::whadup subnet=commune
```

```python 
vali = c.module('vali')
v.serve(tag='whadup', subnet=commune)
v.register(tag='whadup', subnet=commune)
```

This creates a key with "vali::whadup". 

You can also serve it and register it with the following commands:
   
 ```bash
   c serve vali::whadup
   c register vali::whadup
```



to get the validator key is to use the following command:

```bash
c get_key vali::whadup
```


```python
vali = c.module('vali')
vali.get_key('whadup')
```


### Step 2: Staking Your Validator

Ensure that you have staked your validator by following these steps:

1. Stake your validator with another key using the CLI command:

   ```bash
   c stake {validator_key} 20
   ```

   The default amount to be staked is your entire balance. If you don't have a balance, you'll need to unstake.

2. If needed, you can unstake by using the following command:

   ```bash
   c unstake module=vali amount=10
   ```

