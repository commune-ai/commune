
Voting Modules (Validators)

Modules can be anything, but what determines the quality of the network is the validators. Validators are modules that are responsible for voting on the network. They are exactly the same as modules, but they have the ability to vote given enough stake. The minimum stake is determined by the minimum number of allowed weights multiplied by the minimum stake per weight. So if the stake per weight is 100 and the minimum number of votes is 10, you need 1000 tokens to vote at least 10 weights. 

To set weights you can do the following.

```python
# you can use the names, the uids or the keys
c.vote(module=['model.0', 'model.2'], weights=[0,1], netuid=10)
or
c.vote(module=['5E2SmnsAiciqU67pUT3PcpdUL623HccjarKeRv2NhZA2zNES', '5ERLrXrrKPg9k99yp8DuGhop6eajPEgzEED8puFzmtJfyJES'], weights=[0,1], netuid=10)
or 
c.vote(modules=[0, 2], weights=[0,1], netuid=10, key='registered_key_name')
```

The weights can be changed at anytime and are calculated every tempo blocks. 


## Score Function 

The score function can be anything you want it to be. The score function is a function that takes in the weights and outputs a score. The score is then used to determine the trust ratio. The trust ratio is the ratio of the trust score to the stake. The trust score is the number of modules that voted for you. The trust ratio is the trust score divided by the total stake.


## Filter Function 

The filter function is used to filter the names of the modules. This can be used to filter out modules that you do not want to vote on. The filter function is a function that takes in the module name and returns a boolean. If the boolean is true, the module is voted on. If the boolean is false, the module is not voted on. 
To replace the default filter function, you can do the following.

```python

def filter_module(self, module:str):
    """
    Filter Module
    """
    if self.config.search in module:
        return True
    return False

Chain Agnostic Validation over Networks:

Each network is a bundle of modules, which can be refered to as a subnet. This can be onchain (voting enabled) or offchain (no voting enabled for on chain incentives).

A validator can validate the local network, or a blockchain network (subspace, substrate, etc). The validator can also validate multiple networks at once.


How The Validator Works:

The validator scans the network to randomly sample modules within that network. It then calls the score function and returns the score of that functionf for that module. 

If the score is non-zero, the validator will store the module's info and the score. The validator will then vote on the top n modules with the highest scores. 

Validator Voting:

The validator runs a thread in the background that votes periodically based on the vote_staleness. The vote_staleness is the number of blocks before the validator votes. The validator will vote on the top n modules with the highest scores.

Leaderboard

The leaderboard is a list of the top n modules with the highest scores. The leaderboard represents what the validator thinks are the best modules for that score function. The leaderboard can be called from the validator to see the top n modules.







c call vali/leaderboard






