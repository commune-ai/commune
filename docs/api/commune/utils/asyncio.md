# asyncio



Source: `commune/utils/asyncio.py`

## Functions

### `async_read(path, mode='r')`



### `gather(jobs: list, timeout: int = 20, loop=None) -> list`



Type annotations:
```python
jobs: <class 'list'>
timeout: <class 'int'>
return: <class 'list'>
```

### `get_event_loop(nest_asyncio: bool = True) -> 'asyncio.AbstractEventLoop'`



Type annotations:
```python
nest_asyncio: <class 'bool'>
return: asyncio.AbstractEventLoop
```

### `get_new_event_loop(nest_asyncio: bool = False)`



Type annotations:
```python
nest_asyncio: <class 'bool'>
```

### `new_event_loop(nest_asyncio: bool = True) -> 'asyncio.AbstractEventLoop'`



Type annotations:
```python
nest_asyncio: <class 'bool'>
return: asyncio.AbstractEventLoop
```

### `set_event_loop(self, loop=None, new_loop: bool = False) -> 'asyncio.AbstractEventLoop'`



Type annotations:
```python
new_loop: <class 'bool'>
return: asyncio.AbstractEventLoop
```

### `set_nest_asyncio()`



### `sync_wrapper(fn)`



