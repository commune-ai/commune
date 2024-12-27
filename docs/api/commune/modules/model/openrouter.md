# openrouter



Source: `commune/modules/model/openrouter.py`

## Classes

### OpenRouter



#### Methods

##### `ask(self, *args, **kwargs)`



##### `authenticate(self, api_key: str = None, base_url: None = None, timeout: float | None = None, max_retries: int = 5) -> 'OpenAI'`

Authenticate the client with the provided API key, timeout, and max retries.

Args:
    api_key (str): The API key for authentication.
    timeout (float | None, optional): The timeout value for the client. Defaults to None.
    max_retries (int, optional): The maximum number of retries for the client. Defaults to 0.

Type annotations:
```python
api_key: <class 'str'>
base_url: None
timeout: float | None
max_retries: <class 'int'>
return: OpenAI
```

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



##### `forward(self, message: str, *extra_text, history=None, prompt: str = None, system_prompt: str = None, stream: bool = False, model: str = 'claude-3-sonnet', max_tokens: int = 100000, temperature: float = 1.0) -> Union[str, Generator[str, NoneType, NoneType]]`

Generates a response using the OpenAI language model.

Args:
    message (str): The message to send to the language model.
    history (ChatHistory): The conversation history.
    stream (bool): Whether to stream the response or not.
    max_tokens (int): The maximum number of tokens to generate.
    temperature (float): The sampling temperature to use.

Returns:
    Generator[str] | str: A generator for streaming responses or the full streamed response.

Type annotations:
```python
message: <class 'str'>
prompt: <class 'str'>
system_prompt: <class 'str'>
stream: <class 'bool'>
model: <class 'str'>
max_tokens: <class 'int'>
temperature: <class 'float'>
return: typing.Union[str, typing.Generator[str, NoneType, NoneType]]
```

##### `generate(self, message: str, *extra_text, history=None, prompt: str = None, system_prompt: str = None, stream: bool = False, model: str = 'claude-3-sonnet', max_tokens: int = 100000, temperature: float = 1.0) -> Union[str, Generator[str, NoneType, NoneType]]`

Generates a response using the OpenAI language model.

Args:
    message (str): The message to send to the language model.
    history (ChatHistory): The conversation history.
    stream (bool): Whether to stream the response or not.
    max_tokens (int): The maximum number of tokens to generate.
    temperature (float): The sampling temperature to use.

Returns:
    Generator[str] | str: A generator for streaming responses or the full streamed response.

Type annotations:
```python
message: <class 'str'>
prompt: <class 'str'>
system_prompt: <class 'str'>
stream: <class 'bool'>
model: <class 'str'>
max_tokens: <class 'int'>
temperature: <class 'float'>
return: typing.Union[str, typing.Generator[str, NoneType, NoneType]]
```

##### `get_age(self, k: str) -> int`



Type annotations:
```python
k: <class 'str'>
return: <class 'int'>
```

##### `get_key(self)`



##### `get_model_info(self, model)`



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

##### `model2info(self, search: str = None, path='models', max_age=100, update=False)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
search: <class 'str'>
```

##### `model_infos(self, search: str = None, path='models', max_age=0, update=False)`



Type annotations:
```python
search: <class 'str'>
```

##### `models(self, search: str = None, path='models', max_age=60, update=False)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
search: <class 'str'>
```

##### `n(self, search=None)`



##### `n_fns(self, search=None)`



##### `net(self)`



##### `pricing(self, search: str = None, **kwargs)`



Type annotations:
```python
search: <class 'str'>
```

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

##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
```

##### `resolve_model(self, model=None)`



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



