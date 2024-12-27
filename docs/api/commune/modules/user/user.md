# user



Source: `commune/modules/user/user.py`

## Classes

### User



#### Methods

##### `add_admin(self, address)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `add_rate_limit(self, address, rate_limit)`



##### `add_user(self, address, role='user', name=None, **kwargs)`



##### `admins(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `app(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `ask(self, *args, **kwargs)`



##### `blacklist(self)`



##### `blacklist_user(self, address)`



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

##### `df(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

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

##### `get_role(self, address: str, verbose: bool = False)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
address: <class 'str'>
verbose: <class 'bool'>
```

##### `get_user(self, address)`

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

##### `install(self, path)`



##### `is_admin(self, address: str)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
address: <class 'str'>
```

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

##### `is_user(self, address)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

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

##### `refresh_users(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `rm_admin(self, address)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `rm_user(self, address)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `roles(self)`



##### `round(x, sig=6, small_value=1e-09)`



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

##### `setattr(self, k, v)`



##### `sleep(period)`



##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `update_user(self, address, **kwargs)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `user_exists(self, address: str)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
address: <class 'str'>
```

##### `users(self, role=None)`



##### `vs(self, path=None)`



##### `whitelist_user(self, address)`



