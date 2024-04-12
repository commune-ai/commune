
Our goal is to connect all applications into a single decentralized network of things. In doing so we have designed a core abstract called a module.

The Module

A module can represent everything as it is turing complete. This means a module can represent any idea, like an ai model, to a business, to a student loan. In code terms, a module is a simple class in python, where it is a collection of functions that change a state. This general definition is the core foundation of the module

Module Key

Each module is represented as a key which is an sr25519 key. The public key is used to register the module onto the blockchain. The key is used to sign, encrypt,decrypt and verify messages. The key is used to identify the module on the network.


Serving a Module

Serving modules involves converting the python class into an http server. This server only exposes whitelisted functions, and hides powerful functions. You can adjust the whitelist and blacklist functions to your needs. 
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
c serve model.openai::whadup subnet=text
```

Subnets

Subnets are a way to organize modules. Each subnet is a collection of modules that are related to each other. For example, the text subnet can contain modules that are related to text processing. The commune subnet can contain modules that are related to the commune.

To see your existing subnets

c my_subnets

{
    'commune': 0,
    'text': 1
}


** Emission **

Emission is eared when a module votes for you with its stake, or if the module is voting as a validator.

Incentives:

Incentives are rewarded when you are voted for. This includes a module voting for you, or a validator voting for you. The incentive is based on the consensus protocal. Commune has a flexible modular chain that allows for us to adopt different consensus protocals, which include yuma and yomama. This allows us to have a flexible network that can adapt to different use cases.

Conseneus Protocals:

Commune is a flexible modular chain that allows for multiple consensus protocals. The two main protocals are yuma and yomama. Commune intends to have a flexible network that can adapt to different use cases and add additional protocals in the future for different use cases.

**Yuma Voting for Specialized Subnets**

Yuma voting specializes the network to agree by forcing the validators to vote towards the median of the network. This can be good for specialized utility networks, but can also be bad for general networks.

**Yomama Voting for General Subnets**

Yomama voting includes several restrictions to avoid self voting concentrations of power. This can be done through a trust score. The trust score is defined as the number of modules that voted for you. This score is then averaged with the staked voted for you using the trust ratio.

Trust Score = (Number of Modules that voted for you) / (Total Number of Modules)

This allows for a flexible system where the network can decide to be more stake weighted or trust weighted. This allows for a more flexible network that can adapt to different use cases.


Staking 

To stake to a module you need to convert your tokens into the stake onto a module. This stake can then be used by the module to vote. The delegation_fee is the percentage the module gets from the dividends only. If alice has a validator with 10 tokens with a fee of 20 and bob comes in and puts 100 tokens onto alice, alice will recieve 20% + (10 Alice Tokens / 110 Total Tokens) * 80% of the emission. Alice can raise the fee between 5 and 100 percent. Please note that the fee is only taken from the dividends, and not the total emission. 

```python
c.stake('5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES', 100, netuid=10)
```
or to stake multiple amounts to multiple keys, you can do so like this

```python
c.stake_multiple(['5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES', '5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES'], [100, 100], netuid=10)
```

to transfer 100 between two registered modules you can do so like this.

```python
c.transfer_stake('5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES', '5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES', 100, netuid=10)
```

Trustless Emission Profit Sharing 

Each module can profit share its own emission by specifying the fraction of emissions it wants to split across any key. This allows for any module to profit share its emission with any other key (even if its a module).

```python 
keys = ["5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES", "5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES"]
shares = [0.5, 0.5]
c.add_profit_sharing(keys=keys ,shares=shares, netuid=0)
```

Voting on the Network if someone is staked to you, you can vote on the network. This is done by voting for a module. This can be done by voting for a module. 

```python

c.vote(module=['model.0', 'model.2'], weights=[0,1], netuid=10)
```

To vote on the network







