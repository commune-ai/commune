
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

