A subnet is a collection of modules. To register a subnet, you need to specify the subnet parameter to a unique name you want to call the subnet.


The following registers a storage subnet

```bash
c vali register tag=first subnet=storage
```
or 
    
```bash
c register vali::first name=storage
 ```


To create a custom vali, create a module that has a score function.
The score function should return 1 if the module is successful, and 0 if it is not.
You can also create a custom vali that has a score function that returns a score between 0 and 1.

```python
import commune as c

Vali = c.module('vali')

class MyVali(Vali): 
    def __init__(self, **kwargs):
        self.init_vali(**kwargs)
    
    def score(module):
        """
        scores a module 1 if it return successful, and 0 if not
        """

        # this is a virtualized client that connects to the remote module

        info = module.info()
        if isinstance(info, dict):
            return 1
        else:
            return 0

```

## Subnet Parameters

To get the subnet parametes

```bash 
c subnet_params netuid=10
```

When you register a subnet, the default parameters are as such.

tempo (int)
- the number of blocks between each epoch before the votes are calcualted

immunity_period (int)
- the number of blocks before a module can be removed from the subnet due to having the lowest score

min_allowed_weights (int)
- the minimum allowed weights a module can vote.

max_allowed_weights (int)
- the maximum number of weights a module can set during voting.

max_allowed_uids (int)
- the maximum number of modules that can exist on the network.


min_stake (int)
- the minimum stake of a module

founder (str,ss58_address)
- the founder of the subnet, which can change the parameters during authority 

founder_share (int) (min=0, max=100)
- the percentage of the emission the founder gets from the chain

incentive_ratio (int) (min=0, max=100)
- the percentage of emissions that goes towards incentive (miners)

trust_ratio (int) (min=0, max=100)
- the percentage of votes that are weighted as tust (number of modules that vote for you), in addition the voted stake. 
- for example, if 10 of 100 people vote for you and the tust score is 100, then you get 10/100 score. If someone else has twice as many models that vote for them, they get twice the score (20/100). 
    - having a high trust score means that stake is weighted less, but forces people to get the most votes from the network
- if trust is 0, then the votes are entirely based on stake
    - having a low trust means that the votes are more stake based, and are linear with stake. 

vote_threshold (boolean) 
- only applicable during vote_mode=stake

vote_mode (str) Options[authority, stake]
- determines the voting mode for parameter changes of the subnet
- if vote_mode is authority, then the founder can only update the subnet 
- if vote_mode is stake, then a proposal is made and the amount of stake within the subnet needs to reach 50 percent.

self_vote
- self voting allows modules to vote for themselves.

name
- the name of your subnet
    
```bash
{
    'tempo': 1, 
    'immunity_period': 40, 
    'min_allowed_weights': 1,
    'max_allowed_weights': 420,
    'max_allowed_uids': 4096,
    'min_stake': 0,
    'founder': '5DJBFtDLxZ3cahV2zdUzbe5xJiZRqbJdRCdU3WL6txZNqBBj',
    'founder_share': 0,
    'incentive_ratio': 50,
    'trust_ratio': 0,
    'vote_threshold': 50,
    'vote_mode': 'authority',
    'self_vote': True,
    'name': 'subspace',
}

```


# Updating a Subnet (Authority)
As the authority you can redifine any parameter within special global boundaries (see global docs).

As an Authority

The following allows an authority to make a change to the network, you dont even need to specify the key, as it checks if you have the key on your system.
```bash
c update_subnet name=wadup tempo=10 key=authoritykey netuid=10
```



# Updating a Subnet (Authority)

As a democratic system

```bash
c propose_update_subnet name=wadup tempo=10
```

Then to vote for the proposal
```bash
c vote_proposal id=10 key=myvotingkey
```


To unvote a proposal

```bash
c unvote_proposal id=10
```


## Emission Calculations

The calculations are based on the stake that is on your subnet. The more stake on your subnet, the better.




