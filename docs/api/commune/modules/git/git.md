# git



Source: `commune/modules/git/git.py`

## Classes

### git



#### Methods

##### `add_submodule(self, url, name=None, prefix='/home/bakobi/commune/repos')`



##### `ask(self, *args, **kwargs)`



##### `build(self, *args, **kwargs)`



##### `clone(repo_url: str, target_directory: str = None, branch=None)`



Type annotations:
```python
repo_url: <class 'str'>
target_directory: <class 'str'>
```

##### `content(url='LambdaLabsML/examples/main/stable-diffusion-finetuning/pokemon_finetune.ipynb', prefix='https://raw.githubusercontent.com')`



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

##### `get_directory_contents(self, path='')`

Get contents of a directory

##### `get_file_content(self, path)`

Get content of a specific file

##### `get_repos(self, username_or_org='openai')`



##### `get_yaml(path: str = None, default={}, **kwargs) -> Dict`

fLoads a yaml file

Type annotations:
```python
path: <class 'str'>
return: typing.Dict
```

##### `git_repos(self, path='./')`



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

##### `is_repo(self, libpath: str)`



Type annotations:
```python
libpath: <class 'str'>
```

##### `n(self, search=None)`



##### `n_fns(self, search=None)`



##### `net(self)`



##### `print(*text: str, **kwargs)`



Type annotations:
```python
text: <class 'str'>
```

##### `process_code(self, code_content)`

Process the code content - example processing

##### `process_repository(self, path='')`

Process entire repository recursively

##### `progress(*args, **kwargs)`



##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `reset_hard(self, libpath: str = None)`



Type annotations:
```python
libpath: <class 'str'>
```

##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `resolve_libpath(self, libpath: str = None)`



Type annotations:
```python
libpath: <class 'str'>
```

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



##### `vs(self, path=None)`



