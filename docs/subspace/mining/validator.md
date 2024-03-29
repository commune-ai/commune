

## Deploying a Validator Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.

### Step 1: Registering a Validator

To register a validator with a specific tag, use the following CLI command:

```bash
c vali register tag=whadup subnet=commune
or 
c serve vali::whadup # defaults to (netuid=0 subnet=commune key=vali::whadup)
c register vali::whadup # defaults to (netuid=0 subnet=commune key=module)
```

```python
c.serve('vali::whadup')
vali.register('vali::whadup')
```


If you want to register with another key 
   
```bash
c vali register tag=sup subnet=commune key=vali::whadup
```


To check the status of the validator, use the following command:

```bash
c call vali/module_info
# or
c s/get_module vali # or the module name
```

```python
```


```bash
{
    'key': '5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi',
    'name': 'vali',
    'address': '135.180.26.167:50141',
    'emission': 6.440794074,
    'incentive': 0.0,
    'dividends': 0.005676356145571069,
    'last_update': 377547,
    'stake_from': [
        ['5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi', 48.812576334],
        ['5CaWWhTk4D7fphZvjFHKyuaCerqe7uJm3EGNrynGzKczSBNP', 592.049591244],
        ['5ESGbQnTo9RnHEdDpuCYcYDXAAVwHST6dZZ4b5c9JbVr2T3B', 107.994213725],
        ['5EZtFXi8nT6cy55oqCsnsd2Za59PLBVSP9BSjzQEAgh3Pz8M', 0.188784108],
        ['5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC', 403642.241103174],
        ['5EU6rPCkVbPkDJrDhgTmqzu5fpXXKjAdahUYjxKXWZ2U6Q8C', 2.27846803],
        ['5F4sToiPYnbWkg795ryvY5iAVrgDKrpPZv53gaYWEVHHeuKC', 0.002085575],
        ['5CPRaN54kf2cdFauG76kFepE4PeYTc2ttkF4VzF2GCxGaehb', 22322.431257368]
    ],
    'delegation_fee': 50,
    'stake': 426715.998079558
}


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
```
<Keypair (address=5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi, path=vali, crypto_type: SR25519)>
```

or 
   
```bash
c get_key vali::whadup
```



### Staking Your Validator

Ensure that you have staked your validator by following these steps:

1. Stake your validator with another key using the CLI command:

   ```bash
   c stake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200 
   ```
   or 
   ```python
   c.stake("5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi", 200)
   ```
   
   NOTE: The default key is the module key, so you don't need to specify it. If you want to use a different key, you can specify it, as shown in the example above.

   ```bash
   # sends 200 tokens to the validator
   c stake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200 key=vali::whadup
   ```

   The default amount to be staked is your entire balance. If you don't have a balance, you'll need to unstake.

2. If needed, you can unstake by using the following command:

   ```bash
   c unstake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200
   ```

