
# Access Module

**What does this do?** 

This allows servers to control the calls per minute and have them depend on the caller's stake. In this case callers can call a module once per minute per 100 tokens. This prevents dossing from outside miners. 

```
access_module: 
  network: main # mainnet
  netuid: 0 # subnet id
  sync_interval: 1000 #  1000 seconds per sync with the network
  timescale: 'min' # 'sec', 'min', 'hour', 'day'
  stake2rate: 100 # 1 call per every N tokens staked per timescale
  rate: 1 # 1 call per timescale
  fn2rate: {} # function name to rate map, this overrides the default rate

```

**fn2stake**

Lets say if you have a function that is really expensive, and another that just shows the info. Then you want to maybe weight the expensive function more. This access_module allows you to specify functions as follwos

```
fn2rate:
    expensive_function: 1 # 1 call per minute per 100 tokens
    cheap_function: 100 # 100 calls per minute per 100 tokens
```

