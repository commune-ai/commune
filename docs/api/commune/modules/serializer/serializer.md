# serializer



Source: `commune/modules/serializer/serializer.py`

## Classes

### Serializer



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

##### `deserialize(self, x) -> object`

Serializes a torch object to DataBlock wire format.
        

Type annotations:
```python
return: <class 'object'>
```

##### `dict2bytes(self, data: dict) -> bytes`



Type annotations:
```python
data: <class 'dict'>
return: <class 'bytes'>
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

##### `get_data_type_string(self, x)`



##### `get_serializer(self, data_type)`



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

##### `is_repo(self, repo: str)`



Type annotations:
```python
repo: <class 'str'>
```

##### `is_serialized(self, data)`



##### `n(self, search=None)`



##### `n_fns(self, search=None)`



##### `net(self)`



##### `print(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `process_output(self, result, mode='str')`

process the output

##### `progress(*args, **kwargs)`



##### `pull(self)`



##### `push(self, msg: str = 'update')`



Type annotations:
```python
msg: <class 'str'>
```

##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `round(x, sig=6, small_value=1e-09)`



##### `serialize(self, x: dict, mode='dict', copy_value=True)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
x: <class 'dict'>
```

##### `serializer_map(self)`

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

##### `setattr(self, k, v)`



##### `sleep(period)`



##### `str2dict(self, data: str) -> bytes`



Type annotations:
```python
data: <class 'str'>
return: <class 'bytes'>
```

##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `types(self)`



##### `vs(self, path=None)`



