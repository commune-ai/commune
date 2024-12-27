# py



Source: `commune/modules/py/py.py`

## Classes

### Py



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

##### `create(self, env)`



##### `ensure_sys_path()`



##### `enter(self, env)`



##### `env2cmd(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `env2libs(self)`



##### `env2path(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `envs(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `envs_paths(self)`



##### `epoch(self, *args, **kwargs)`



##### `file2hash(self, path='./')`



##### `fn_n(self, search=None)`



##### `forward(self, *args, **kwargs)`



##### `get_activation_path(self, env)`



##### `get_age(self, k: str) -> int`



Type annotations:
```python
k: <class 'str'>
return: <class 'int'>
```

##### `get_env(self, env)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

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

##### `install(self, env, package_name)`



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

##### `libs(self, env=None, search=None)`

Available environments:

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

##### `remove(self, env)`



##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `round(x, sig=6, small_value=1e-09)`



##### `run(self, path='/home/bakobi/commune/modules/sandbox.py', env='bt')`



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

##### `set_venv_path(self, venv_path)`



##### `setattr(self, k, v)`



##### `sleep(period)`



##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `vs(self, path=None)`



