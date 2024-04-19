




Th


**Stake Based Conseneus Protocals**

Commune is a flexible modular chain that allows for multiple consensus protocals. The two main protocals are yuma and yomama. Commune intends to have a flexible network that can adapt to different use cases and add additional protocals in the future for different use cases.

**Linear**

Linear is the simplest in that it represents a linear distribution of the rewards. This is good for a general network that does not require any specialized voting. The downside is that it can be easily manipulated by cabals or dishonest voting. This requires additional security measures to prevent dishonest voting.

**Yuma**
Yuma specializes the network to agree by forcing the validators to vote towards the median of the network. This can be good for specialized utility networks and commune has this an an option. The whole thesis of yuma is to incentivize intelligence without dishonest voting or cabals voting for themselves. 

**Yomama**
Yomama voting includes several restrictions to avoid self voting concentrations of power as does yuma. This can be done through a trust score. The trust score is defined as the number of modules that voted for you. This score is then averaged with the staked voted for you using the trust ratio. 

Trust Score = (Number of Modules that voted for you) / (Total Number of Modules)

This allows for a flexible system where the network can decide to be more stake weighted or trust weighted. This allows for a more flexible network that can adapt to different use cases.


**yumaswap**

The yumaswap protocal uses the uniswap protocal to incentivize early stakers more than later stakers. This is done by using a bonding curve to calculate the rewards. The bonding curve is defined as the following:

```python
reward = emission * (1 - delegation_fee) * (stake / total_stake)
```

The reward is then added to the stake of the module. The total stake is the sum of all stakes on the module. The delegation_fee is the fee the module takes from the reward. The emission is the total emission of the module.




