# executor



Source: `commune/executor.py`

## Classes

### Future

Represents the result of an asynchronous computation.

#### Methods

##### `add_done_callback(self, fn)`

Attaches a callable that will be called when the future finishes.

Args:
    fn: A callable that will be called with this future as its only
        argument when the future completes or is cancelled. The callable
        will always be called by a thread in the same process in which
        it was added. If the future has already completed or been
        cancelled then the callable will be called immediately. These
        callables are called in the order that they were added.

##### `cancel(self)`

Cancel the future if possible.

Returns True if the future was cancelled, False otherwise. A future
cannot be cancelled if it is running or has already completed.

##### `cancelled(self)`

Return True if the future was cancelled.

##### `done(self)`

Return True if the future was cancelled or finished executing.

##### `exception(self, timeout=None)`

Return the exception raised by the call that the future represents.

Args:
    timeout: The number of seconds to wait for the exception if the
        future isn't done. If None, then there is no limit on the wait
        time.

Returns:
    The exception raised by the call that the future represents or None
    if the call completed without raising.

Raises:
    CancelledError: If the future was cancelled.
    TimeoutError: If the future didn't finish executing before the given
        timeout.

##### `result(self, timeout=None)`

Return the result of the call that the future represents.

Args:
    timeout: The number of seconds to wait for the result if the future
        isn't done. If None, then there is no limit on the wait time.

Returns:
    The result of the call that the future represents.

Raises:
    CancelledError: If the future was cancelled.
    TimeoutError: If the future didn't finish executing before the given
        timeout.
    Exception: If the call raised then that exception will be raised.

##### `running(self)`

Return True if the future is currently executing.

##### `set_exception(self, exception)`

Sets the result of the future as being the given exception.

Should only be used by Executor implementations and unit tests.

##### `set_result(self, result)`

Sets the return value of work associated with the future.

Should only be used by Executor implementations and unit tests.

##### `set_running_or_notify_cancel(self)`

Mark the future as running or process any cancel notifications.

Should only be used by Executor implementations and unit tests.

If the future has been cancelled (cancel() was called and returned
True) then any threads waiting on the future completing (though calls
to as_completed() or wait()) are notified and False is returned.

If the future was not cancelled then it is put in the running state
(future calls to running() will return True) and True is returned.

This method should be called by Executor implementations before
executing the work associated with this future. If this method returns
False then the work should not be executed.

Returns:
    False if the Future was cancelled, True otherwise.

Raises:
    RuntimeError: if this method was already called or if set_result()
        or set_exception() was called.

### Task



#### Methods

##### `cancel(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `done(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

##### `result(self) -> object`



Type annotations:
```python
return: <class 'object'>
```

##### `run(self)`

Run the given work item

##### `running(self) -> bool`



Type annotations:
```python
return: <class 'bool'>
```

### ThreadPoolExecutor

Base threadpool executor with a priority queue

#### Methods

##### `adjust_thread_count(self)`



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

##### `default_priority_score(self)`



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

##### `repo2path(self, search=None)`



##### `repos(self, search=None)`



##### `resolve_key(self, key: str = None) -> str`



Type annotations:
```python
key: <class 'str'>
return: <class 'str'>
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



##### `shutdown(self, wait=True)`



##### `sleep(period)`



##### `status(self)`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

##### `submit(self, fn: Callable, args: dict = None, kwargs: dict = None, params=None, priority: int = 1, timeout=200, return_future: bool = True, wait=True, path: str = None) -> concurrent.futures._base.Future`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
fn: typing.Callable
args: <class 'dict'>
kwargs: <class 'dict'>
priority: <class 'int'>
return_future: <class 'bool'>
path: <class 'str'>
return: <class 'concurrent.futures._base.Future'>
```

##### `sync(self)`



##### `syspath(self)`



##### `time(self)`



##### `tqdm(*args, **kwargs)`



##### `vs(self, path=None)`



##### `wait(futures: list) -> list`

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

Type annotations:
```python
futures: <class 'list'>
return: <class 'list'>
```

##### `worker(executor_reference, work_queue)`



