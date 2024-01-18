

## Deploying a Validator Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.



### Step 1: Registering a Validator

To register a validator with a specific tag, use the following CLI command:

```bash
c vali register tag=whadup
```

```python 
c.module('vali').register(tag='whadup')
```

This creates a key with "vali::whadup". 

You can also serve it and register it with the following commands:
   
 ```bash
   c serve vali::whadup
   c register vali::whadup
```


### Step 2A: Staking Your Validator

Ensure that you have staked your validator by following these steps:

1. Stake your validator with another key using the CLI command:

   ```bash
   c stake vali::whadup {amount}
   ```

   The default amount to be staked is your entire balance. If you don't have a balance, you'll need to unstake.

2. If needed, you can unstake by using the following command:

   ```bash
   c unstake {amount}
   ```


### Step 2B: Staking to Multiple Validators

You can stake to multiple validators by using the following command:


Stake 100 form your balance to vali::whadup and vali::whadup2

```bash
c stake_many amount=100 modules=["vali::whadup","vali::whadup2"]
```


### Step 2C: Stake Spread

You can stake to multiple validators by using the following command:


Stake 100 form your balance to vali::whadup and vali::whadup2

```bash
c stake_spread whadup amount=100
```



## Getting Your Validator's Status

To get your validator's status, use the following command:

```bash
c vstats
```

```python
c.vstats()
```

![Alt text](image.png)
output



