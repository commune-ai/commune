# AsyncioThreadExecutor Class

This script brings together Python's built-in libraries, `asyncio` and `concurrent.futures`, to create an instance of a threaded asynchronous executor.

## Features

1. **Concurrency**: This script uses Python's `concurrent.futures.ThreadPoolExecutor` to handle and create multiple threads, thus enabling parallel execution of tasks.

2. **Exception Handling**: It catches exceptions raised during the execution of tasks, prints an error message, sets the `_shutdown` flag to `True`, and re-raises the exception.

3. **Asynchronous Task Execution**: This script uses asyncio to run-in-executor the tasks that are to be executed by the thread pool executor. In addition, it supports scheduling of tasks using asyncio on the event loop, enabling concurrency of tasks even on a single thread.

## Installation & Usage

To use, simply import the module at the top of your Python script:

```python
import concurrent
import asyncio
```
An instance of the `AsyncioThreadExecutor` can be created by calling the class as below:

```python
executor = AsyncioThreadExecutor(max_threads=10)
```
You may call `executor.run(fn, tasks)` to start executing tasks in the thread pool. Here `fn` is typically a function, and `tasks` is a list of arguments to be passed to `fn`.

## Example

```python
def fn(*args):
    # function to be executed in thread
    ...

executor = AsyncioThreadExecutor(max_threads=10)
tasks = [[args1], [args2]]
executor.run(fn, tasks) # starts executing tasks in thread pool
```

## Contributing
We welcome bug reports, feature requests, and contributions. To report a bug or request a feature, please open an issue on our GitHub page. For direct contributions, please fork the repository, make your changes, and open a pull request.
