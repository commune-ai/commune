to start the subnet


c subnet/testnet valis=2 miners=3

this creates a testnet on your computer 
[
    {'success': True, 'name': 'subnet.miner::0', 'address': '100.38.7.242:50125', 'kwargs': {}},
    {'success': True, 'name': 'subnet.miner::1', 'address': '100.38.7.242:50158', 'kwargs': {}},
    {'success': True, 'name': 'subnet.miner::2', 'address': '100.38.7.242:50161', 'kwargs': {}},
    {'success': True, 'name': 'subnet.vali::0', 'address': '100.38.7.242:50086', 'kwargs': {}}
]


c servers

[
    'subnet.miner::0',
    'subnet.vali::0'
]


c call subenet.vali::0/leaderboard

```
  __import__('pkg_resources').require('commune==0.0.1')
               name    w  staleness   latency                                      ss58_address
0  model.openrouter  0.0   0.707295  0.000178  5ECqPjZD7TmWPjoy6Z9n21qpScR3EpGBUysdpaMruU6Z1npW
1            module  0.0   0.079662  0.000188  5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC
2           storage  0.0   1.232370  0.000195  5H6P4f9VoFLSsuPnq6KtEbG9VTEP2oTddXEHaJ4BDz9GXUgi
3   subnet.miner::0  0.0   0.809937  0.000213  5H3nc7FGi2fvK5H6Yq7QfwmWT59Cb8PeGpP6fA64QfoTrBTk
4    subnet.vali::0  0.0   0.603564  0.000164  5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC
5          subspace  0.0   0.395964  0.000302  5HLA9NhuV5bjNur5qBAFo6CtXmn945FYYv3sTueQ5PHBg98X

```

register onchain 

c.register("subenet.vali::0", subnet="commune")


