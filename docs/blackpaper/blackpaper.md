
I am not the best at writing as I see ideas and its hard for me to put them down, unless i spend time. But here is a recap of commune. Our goal is to connect all applications into a decentralized network. This involves creating a tokenomics structure that can adapt to everything by providing incentives to run applicaitons. To do this, we required to 

The Module

A module can represent everything as it is turing complete. This means a module can represent any idea, like an ai model, to a business, to a student loan. In code terms, a module is a simple class in python, where it is a collection of functions that change a state. This general definition is the core foundation of the module.

The Module is Simple and Complex? 

Like a python class (to whic we try to maximally mimic as much as possible), a module can represent a simple idea like a function, or a complex idea like a business. This allows for a flexible network that can adapt to different use cases. The module is the core abstract that allows for a flexible network that can adapt to different use cases. This is becuase the module is turing complete, and can represent any idea.

Module Key

Each module is represented as a key which is an sr25519 key. The public key is used to register the module onto the blockchain. The key is used to sign, encrypt,decrypt and verify messages. These keys can also represent other keys on any other chain through transfering its seed entropy to another chain key format. This allows for modules to exist on any chain

For Warning for Anti-Python Peeps

The module is designed from a python class, but this can be implemented in rust, javascript, or any other language, even if they are functional programming languages (via structs). You can think of the module as a class in any language, where it is a collection of functions that change a state. This is the core foundation of the module.

This is a simple example of a module that says whadup. 

```python
import commune as cs
class Model(c.Module):
    def __init__(self, c=0):
        self.c = c
    
    def add(self, a=1, b=1):
        return a + b + self.c

```

I can serve this as an api which runs in the background 

c.serve(Whadup, name="whadup_dawg")

calling whadup/ function and it will return a + b + c as defined inside the function. 


Serving a Module

Serving modules involves converting the python class into an http server. This server only exposes whitelist and blacklist functions, and hides powerful functions. You can adjust the whitelist and blacklist functions to your needs. 
When you serve the module, you will need to give it a name. By default it is 

```bash
c serve model.openai::whadup
```

This will serve the module onto the network. To register it onto the blockchain, you will need to register it. 

```
c register model.openai::whadup # defults to subnet=commune
```

This will register the module onto the blockchain. If the subnet name is not provided, it will default to the commune.
if you want to serve the module on a different subnet, you can do so by providing the subnet name like so.

```bash
c register model.openai::whadup subnet=text
```

Namespaces

Namespaces are a way to organize modules. Each namespace is a collection of modules that associates. To see the namespace of a network

```python
c.namespace(network='local')
```
```bash
{'subnet.add.subnet.vali': '0.0.0.0:50214', 'subnet.vali::0': '0.0.0.0:50086', 'vali': '0.0.0.0:50204'}
```

Subnets

Subnets are a way to organize modules on chain. Each subnet is a collection of modules that are related to each other. For example, the text subnet can contain modules that are related to text processing. The commune subnet can contain modules that are related to the commune.

To create your subnet you can do so like this. This registers a subnet onto the chain.

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

**Emission**

Emission is eared every tempo blocks which is set per subnet. 

**Incentives**
Incentives are rewarded when you are voted for. This includes a module voting for you, or a validator voting for you. The incentive is based on the consensus protocal. Commune has a flexible modular chain that allows for us to adopt different consensus protocals, which include yuma and yomama. This allows us to have a flexible network that can adapt to different use cases.

**Dividends**
Dividends are rewarded to the modules that are voting. The delegation_fee parameter is a module parameter that defines the percentage the module owner gets. The minimum and maximum is 5 to 100 percent. The rest of the dividends are distributed based on the stake.

Voting Modules (Validators)

Modules can be anything, but what determines the quality of the network is the validators. Validators are modules that are responsible for voting on the network. They are exactly the same as modules, but they have the ability to vote given enough stake. The minimum stake is determined by the minimum number of allowed weights multiplied by the minimum stake per weight. So if the stake per weight is 100 and the minimum number of votes is 10, you need 1000 tokens to vote at least 10 weights. 

To set weights you can do the following.

```python
# you can use the names, the uids or the keys
c.vote(module=['model.0', 'model.2'], weights=[0,1], netuid=10)
or
c.vote(module=['5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES', '5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES'], weights=[0,1], netuid=10)
or 
c.vote(module=[0, 2], weights=[0,1], netuid=10)
```

The weights can be changed at anytime and are calculated every tempo blocks. 


Trustless Emission Profit Sharing 

Each module can profit share its own emission by specifying the fraction of emissions it wants to split across any key. This allows for any module to profit share its emission with any other key (even if its a module).

```python 
keys = ["5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES", "5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES"]
shares = [0.5, 0.5]
c.add_profit_sharing(keys=keys ,shares=shares, netuid=0)
```

Voting on the Network if someone is staked to you, you can vote on the network. This is done by voting for a module. This can be done by voting for a module. 

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


