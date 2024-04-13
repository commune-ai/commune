Subnets

Subnets are a way to organize modules on chain. Each subnet is a collection of modules that are related to each other. For example, the text subnet can contain modules that are related to text processing. The commune subnet can contain modules that are related to the commune.

To create your subnet you can do so like this. This registers a subnet onto the chain.
x
```python
c.register('vali::subnet', kwargs={'subnet': subnet},  subnet='text')
```

If the maximum subnets are reached, the chain will remove the least staked subnet. The owner can change the parameters of the subnet.
These parameters include 

```python
{
    'founder': '5HarzAYD37Sp3vJs385CLvhDPN52Cb1Q352yxZnDZchznPaS',
    'min_allowed_weights': 256, # the minimum amount of weights required to create a subnet
    'immunity_period': 1000, # the period of time for a module to be on the network before it can be removed
    'vote_mode': 'Authority', # the vote mode can be Authority or Democracy
    'min_stake': 256.0, # the minimum stake required to create a subnet
    'max_stake': 1000000.0, # the maximum stake allowed per module
    'max_allowed_uids': 8144, # the maximum amount of uids allowed to create a subnet
    'max_allowed_weights': 512, # the maximum amount of weights allowed to create a subnet 
    'max_weight_age': 1000, # the maximum age of the weights (0-2^32)
    'founder_share': 0,    # the share of the founder (0-100)
    'incentive_ratio': 50, # the ratio of the incentive (0-100)
    'name': 'commune', # the name of the subnet (0-100)
    'tempo': 100, # the number of blocks before calculating the votes
    'trust_ratio': 20 # the trust ratio of the subnet (0-100)
}
```

The subnet can be created by the founder. The founder can be changed by the founder. The founder can also change the parameters of the subnet.



## Updating a Subnet

To Update a Subnet




