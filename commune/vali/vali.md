


# Validator

This validator is the foundation of on-chain and off-chain evaluations over any server/module. This reduces on-chain leaderboards from excluding off-chain members, by allowing anyone to create a validator
The rules are simple

```
import commune as c
Vali = c.module('vali')
class AddVali(Vali):
    def __init__(search='add', network='local', **kwargs):
        self.init_vali(locals())
    def score(self, module, a=1, b=2):
        return int(module.forward(a=a, b=b) == (a + b))
```


### Network

To set the network, the validator uses the set_network/sync/sync_network (all alias to the same thing) function. This function sets the network to the network specified in the function. The network has a namespace which maps names->addresses. The network can be onchain or offchain, and commune lets you specify it with the fn:set_network function.

The following example allows us to switch to subspace (onchain) networks after serving the validator on local (offchain).

```python
# sets the network to local at default
c.serve('vali', network='local') 

# changes the network to subspace, the backend will then adjust accordingly
c.connect('vali').set_network(network='subspace', netuid=0, search='model')

```

The validator applies a **fn:score** function to the modules over a network's namespace, which is a map of names->addresses. 
Namespaces can be onchain or offchain, and commune lets you specifiy it with the fn:set_network function.

the score function can be defined by simply specifying the score function in the validator. 

The namespace can be filtered by the **search** field, which includes modules with the name of the search field. For instance, search=model filters 
modules that have model in the name. This is useful for filtering out modules that are not relevant to the validator. You can also use custom search functions and can filter the namespace to whatever you want.


### Run Loop Thread

When the Network is set, the run_loop is started as a background thread

c.thread(self.run_loop)

What does the run_loop do?

- starts the workers as background threads, (config.workers = 1).
- votes periodically over the network (only for voting networks)
- Syncs with the network periodically (config.sync_interval)


### Worker/Epoch Thread

The worker threads run epochs in a forever loop. The epoch randomly runs through the namespapce and evaluates the modules in the namespace.
Each worker can be a thread, a process or a server. This is defined by the config.mode. Note that only thread is supported, but we will support process and server modes.


```python
def epoch(self):
    module_addresses = list(self.namespace.values())
    for module_address in module_addresses
        result = self.eval(module_address)
        results += [result]
    return results

```


## Deploying a Validator Tutorial on Subspace

In this tutorial, you will learn how to deploy a validator on the network and perform various tasks related to staking, registration, and validation.
This will involve a custom validator that is inherited from the "vali" module. To check more about the 

Run the validator on the subspace network

```python
import commune as c
c.serve('vali.text.realfake::tang', network='subspace', netuid=0)
```

Ensure you have the module registered on the network
```python
c.register('vali.text.realfake::tang', netuid=0, stake=10000)
assert c.is_registered('vali.text.realfake::tang', netuid=0) == True
```


### Leaderboard

Once your validator is running, you need to make sure you have a leaderboard

```python
c.connect('vali.text.realfake::tang').leaderboard()
# or
c.call('vali.text.realfake::tang/leaderboard')
```

```python
                             name    w   staleness   latency
0           model.bitapi::aaron65  0.5  169.770592  0.442878
364      model.openai::opiod22404  0.5    2.788784  0.336754
363      model.openai::opiod19132  0.5  156.788734  0.292289
362       model.openai::opiod1657  0.5    7.788687  0.355320
361      model.openai::opiod14444  0.5  112.788638  0.841325
..                            ...  ...         ...       ...
537  model.openrouter::neutrino.9  1.0  164.798043  0.779351
536  model.openrouter::neutrino.8  1.0  162.797987  0.222788
535  model.openrouter::neutrino.7  1.0  119.797929  0.172207
533  model.openrouter::neutrino.5  1.0   57.797815  0.921272
418         model.openai::virago1  1.0  149.791477  1.435314

[624 rows x 4 columns]

```

Note: If you get 0 rows, then you need to make sure your requests are being saved and running properly. This can be a result of a broken epoch where requests are not being saved due to an error. 


Run Info:

The run info is important for debugging a validator, and ensuring it is sending requests, voting, and functioning propery


The run info combines all of the network_info, epoch_info, and vote_info functions as follows.

```python
{
    'network': {
        'search': 'model', # search term that filters modules based on ones that have "model" in it
        'network': 'subspace', # the name of the network (local, subspace)
        'netuid': 0, # the netuid
        'n': 1750, # the number of modules in the network
        'fn': None, # the function to be called (OPTIONAL)
        'staleness': 715.3814945220947 # the last time the module was synced
    },
    'epoch': {
        'requests': 2400, # requests sent
        'errors': 19, # errors
        'successes': 2380, # number of successes
        'sent_staleness': 0.0753633975982666, # time since a REQUEST was sent (if this is too high, your epoch has stalled)
        'success_staleness': 0.4050748348236084, # time since a SUCCESS (w>0) was completed (if this is too high, your epoch has stalled)
        'staleness_count': 0, #
        'epochs': 0,
        'executor_status': {
            'num_threads': 64, # number of threads
            'num_tasks': 0, # number of tasks in the queue
            'is_empty': True, # is the executor empty?
            'is_full': False # is the executor full?
        }
    },
    'vote': {
        'num_uids': 109, # number of uids votedf or
        'staleness': 84, # staleness of the votes in blocks 
        'key': '5H1T1YFxw6CThLbfQttmcdokaVWfgfCS79rtcjFqF2HWbvtP', # key used to vote (the validator key)
        'network': 'subspace' # the voting network
    }
}

```



Module Info for Information of the Validator

c.module_info('KEYORNAME', netuid=0)

To check the status of the validator, use the following command:

```python
c.call("vali.text.realfake::sup/module_info")
```

```bash
{
    'key': '5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi', # key of the validator
    'name': 'vali.text.realfake::sup', # name of the module
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

```   



### Staking Your Validator

Ensure that you have staked your validator by following these steps:

1. Stake your validator with another key using the CLI command:


   ```bash
   # stake 200 tokens to the validator using the key=module (default key)
   c stake 5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi 200 key=module
   ```
   or 
   ```python
   c.stake("5GN545yeaTEuzDEoD6EXPi4qFqQCABKNnsUyJfDHFYWzfmQi", amount=200, netuid=0, key='module')
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


###  Updating Your Validator


To update your validator, use the following command:

You can update your module's following parameters:

- `delegation_fee`: The delegation fee for the validator.
- `name`: The name of the validator.
- `address`: The address of the validator.
- `metadata`: The metadata 


The default is used if the parameter is not provided from the module_info
To check the module info check the following.


```python

# If we want to update the delegation fee to 10, and the name to vali::whadup2 and an address with 123.435.454:6969:
c.update_module(module='vali::sup', delegation_fee=10, name='vali::whadup2', address=123.435.454:6969)

```

Updating Multiple Validators at Once

Sometimes your validators go offline and reserve on a different port. This is not common but when it happens, you can update all of the servers at once using the following command:

The following command updates all of the servers at once:

```python
c.update_modules(search='vali')
```

To update multiple validators at once to a delegation_fee of 10 , use the following command:

```python

c.update_modules(search='vali', delegation_fee=10)

```



# Debugging

If you fucked up, then you should enable debug mode to see the errors

c.serve('vali.text.realfake::tang', debug=1)

- Make sure the leaderboard is not empty




