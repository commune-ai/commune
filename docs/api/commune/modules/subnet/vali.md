# vali



Source: `commune/modules/subnet/vali.py`

## Classes

### Vali



#### Methods

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



##### `epoch(self)`



##### `file2hash(self, path='./')`



##### `fn_n(self, search=None)`



##### `forward(self, *args, **kwargs)`



##### `get_age(self, k: str) -> int`



Type annotations:
```python
k: <class 'str'>
return: <class 'int'>
```

##### `get_client(self, module: dict)`



Type annotations:
```python
module: <class 'dict'>
```

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

##### `init_vali(self, network='local', subnet: Union[str, int, NoneType] = None, search: Optional[str] = None, batch_size: int = 128, max_workers: Optional[int] = None, score: Union[ForwardRef('callable'), int] = None, key: str = None, path: str = None, tempo: int = None, timeout: int = 3, update: bool = False, run_loop: bool = True, **kwargs)`

Initialize self.  See help(type(self)) for accurate signature.

Type annotations:
```python
subnet: typing.Union[str, int, NoneType]
search: typing.Optional[str]
batch_size: <class 'int'>
max_workers: typing.Optional[int]
score: typing.Union[ForwardRef('callable'), int]
key: <class 'str'>
path: <class 'str'>
tempo: <class 'int'>
timeout: <class 'int'>
update: <class 'bool'>
run_loop: <class 'bool'>
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

##### `is_repo(self, repo: str)`



Type annotations:
```python
repo: <class 'str'>
```

##### `module_paths(self)`



##### `n(self, search=None)`



##### `n_fns(self, search=None)`



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

##### `refresh_scoreboard(self)`



##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `round(x, sig=6, small_value=1e-09)`



##### `run_loop(self)`



##### `score(self, module)`



##### `score_module(self, module: dict, **kwargs)`

module: dict
    name: str
    address: str
    key: str
    time: int

Type annotations:
```python
module: <class 'dict'>
```

##### `score_modules(self, modules: List[dict])`



Type annotations:
```python
modules: typing.List[dict]
```

##### `scoreboard(self, keys=['name', 'score', 'latency', 'address', 'key'], ascending=True, by='score', to_dict=False, page=None, **kwargs)`



##### `set_config(self, config: Union[str, dict, NoneType] = None) -> 'Munch'`

Set the config as well as its local params

Type annotations:
```python
config: typing.Union[str, dict, NoneType]
return: Munch
```

##### `set_key(self, key)`



##### `set_network(self, network: str, subnet: str = None, tempo: int = 60, search: str = None, path: str = None, score=None, update=False)`



Type annotations:
```python
network: <class 'str'>
subnet: <class 'str'>
tempo: <class 'int'>
search: <class 'str'>
path: <class 'str'>
```

##### `set_score(self, score)`



##### `setattr(self, k, v)`



##### `sleep(period)`



##### `sync(self, update=False)`



##### `syspath(self)`



##### `test(n=2, tag='vali_test_net', miner='module', trials=5, tempo=4, update=True, path='/tmp/commune/vali_test', network='local')`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `vote(self, results)`



##### `vs(self, path=None)`



