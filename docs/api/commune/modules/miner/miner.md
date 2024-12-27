# miner



Source: `commune/modules/miner/miner.py`

## Classes

### Miner



#### Methods

##### `add_keys(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `ask(self, *args, **kwargs)`



##### `build(self, *args, **kwargs)`



##### `clone(self, repo: str, path: str = None, **kwargs)`



Type annotations:
```python
repo: <class 'str'>
path: <class 'str'>
```

##### `copy_module(self, module: str, path: str)`



Type annotations:
```python
module: <class 'str'>
path: <class 'str'>
```

##### `ensure_sys_path()`



##### `epoch(self, *args, **kwargs)`



##### `file2hash(self, path='./')`



##### `fn_n(self, search=None)`



##### `forward(self, *args, **kwargs)`



##### `get_age(self, k: str) -> int`



Type annotations:
```python
k: <class 'str'>
return: <class 'int'>
```

##### `get_miner_name(self, key)`



##### `get_yaml(path: str = None, default={}, **kwargs) -> Dict`

fLoads a yaml file

Type annotations:
```python
path: <class 'str'>
return: typing.Dict
```

##### `has_module(self, path: str)`



Type annotations:
```python
path: <class 'str'>
```

##### `install(self, path)`



##### `is_encrypted(self, path: str) -> bool`



Type annotations:
```python
path: <class 'str'>
return: <class 'bool'>
```

##### `is_error(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `is_registered(self, key)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `is_repo(self, repo: str)`



Type annotations:
```python
repo: <class 'str'>
```

##### `is_running(self, name)`



##### `key2exist(self)`



##### `key_addresses(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `key_names(self)`



##### `key_state(self)`



##### `keys(self, names=False)`



##### `kill_miner(self, name)`



##### `leaderboard(self, avoid_keys=['stake_from', 'key', 'vote_staleness', 'last_update', 'dividends', 'delegation_fee'], sort_by='emission', reverse=False)`



##### `load_keys(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `miner2balance(self, timeout=30, max_age=30, **kwargs)`



##### `modules(self, max_age=600, update=False, **kwargs)`



##### `n(self, search=None)`



##### `n_fns(self, search=None)`



##### `names(self)`



##### `net(self)`



##### `print(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `progress(*args, **kwargs)`



##### `pull(self)`



##### `push(self, msg: str = 'update')`



Type annotations:
```python
msg: <class 'str'>
```

##### `register_keys(self, timeout=60, parallel=False, controller=None)`



##### `register_miner(self, key, controller=None, stake=1, nonce=None)`



##### `register_miners(self, timeout=60, parallel=False, controller=None)`



##### `registered_keys(self, **kwargs)`



##### `rename_keys(self, new_prefix)`



##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_controller(self, controller)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `resolve_key_address(self, key)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `resolve_keys(self)`



##### `round(x, sig=6, small_value=1e-09)`



##### `run_miner(self, key, refresh=False)`



##### `run_miners(self, refresh=False, **kwargs)`



##### `save_keys(self, path='miner_mems')`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `servers(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `set_config(self, config: Union[str, dict, NoneType] = None) -> 'Munch'`

Set the config as well as its local params

Type annotations:
```python
config: typing.Union[str, dict, NoneType]
return: Munch
```

##### `set_key(self, key: str, **kwargs) -> None`



Type annotations:
```python
key: <class 'str'>
return: None
```

##### `set_subnet(self, netuid=None, max_age=10000, update=False, **kwargs)`



##### `setattr(self, k, v)`



##### `sleep(period)`



##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `transfer_to_miners(self, amount)`



##### `uids(self)`



##### `unregisered_keys(self)`



##### `unstake_and_transfer_back(self, key, amount=20)`



##### `unstake_many(self, amount=50, transfer_back=True)`



##### `vs(self, path=None)`



