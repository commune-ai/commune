to start the subnet


c subnet/testnet 

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


salvivona@Sals-MacBook-Pro commune % c call subnet.vali::0/config     
/opt/homebrew/bin/c:4: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  __import__('pkg_resources').require('commune==0.0.1')
{
    'network': 'local',
    'netuid': 0,
    'verbose': False,
    'sync_interval': 10,
    'min_update_interval': 1,
    'sleep_interval': 5,
    'sample_sleep_interval': 0.1,
    'initial_sleep': 1,
    'search': None,
    'max_age': 3600,
    'vote': True,
    'fn': None,
    'alpha': 0.5,
    'worker_fn_name': 'worker',
    'min_stake': 1,
    'vote_interval': 100,
    'vote_timeout': 50,
    'voting_networks': ['subspace', 'bittensor'],
    'max_history': 10,
    'connect_score': 0.1,
    'latency_score_weight': 0.2,
    'vote_tag': None,
    'mode': 'thread',
    'batch_size': 32,
    'workers': 1,
    'threads_per_worker': 32,
    'timeout': 3,
    'sleep_time': 0.05,
    'refresh': True,
    'start': True,
    'is_main_worker': True,
    'min_num_weights': 10,
    'debug': False,
    'print_interval': 2,
    'clone_suffix': 'clone',
    'tag': '0',
    'server_name': 'subnet.vali::0'
}



