

## Deploying a Validator Tutorial

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.

### Step 1: Registering a Validator

To register a validator with a specific tag, use the following CLI command:


```python
import commune as c
c.serve('vali::whadup')
vali.register('vali::whadup')
```

```bash
c serve vali::whadup network=subspace # deploys a validator to review the network
c register vali::whadup # defaults to (netuid=0 subnet=commune key=module)
```

If you want to register with another key 
   
```bash
c register vali netuid=0 key=vali::whadup
```


To check the status of the validator, use the following command:

```bash
c call vali/module_info
```

```python
c.call("vali/module_info")
```

```bash
{
    'key': '5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi', # key of the validator
    'name': 'vali', # name of the module
    'address': '135.180.26.167:50141', # address of the validator
    'emission': 6.440794074, # emission to the validator
    'incentive': 0.0, # incentives to the validator
    'dividends': 0.005676356145571069, # dividends (rewards) to the validator
    'last_update': 377547, # last update of the weights 
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
    'delegation_fee': 50, # delegation fee
    'stake': 426715.998079558 # total stake to the module
}

### Staking Your Validator

Ensure that you have staked your validator by following these steps:

1. Stake your validator with another key using the CLI command:


   ```bash
   # stake 200 tokens to the validator using the key=module (default key)
   c stake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200 key=module
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
   c unstake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200 key=module
   ```
   or
    ```python
    c.unstake("5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi", 200, key="module")
    ```


### Step 3: Updating Your Validator


To update your validator, use the following command:

You can update your module's following parameters:

- `delegation_fee`: The delegation fee for the validator.
- `name`: The name of the validator.
- `address`: The address of the validator.


```bash

c update_module vali::whadup delegation_fee=10 name=vali::whadup2

``


Updating Multiple Validators at Once

Sometimes your validators go offline and reserve on a different port. This is not common but when it happens, you can update all of the servers at once using the following command:


The following command updates all of the servers at once:
```bash
c update_modules search=vali
```

To update multiple validators at once to a delegation_fee of 10 , use the following command:

```bash

c update_modules search=vali delegation_fee=10 

```




