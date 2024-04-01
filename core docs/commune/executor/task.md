# Task Module

Task is a Python class designed to provide advanced task handling capabilities in Python. It allows developers to handle asynchronous tasks, schedule task execution, and handle task timeouts effectively. 

## Class Attributes

- `future`: An instance of concurrent.futures.Future that represents a computation that hasn't necessarily completed yet.
- `fn`: The function to be executed when running the task.
- `start_time`: The time when the task was created.
- `args`: The arguments to pass to the function `fn` when it is run.
- `kwargs`: The keyword arguments to pass to the function `fn` when it is run.
- `timeout`: The time within which the function `fn` must complete execution.
- `priority`: The priority assigned to the task.
- `data`: The result of running the function `fn`.
- `extra_kwargs`: Additional keyword arguments.
- `save`: A boolean value indicating whether to save the task state or not.
- `status`: The current status of the task. It can be 'pending', 'running', or 'done'.

## Class Methods

- `lifetime`: Returns the time elapsed since the task was created.
- `state`: Returns a dictionary containing the state of the task.
- `save_state`: Saves the task state.
- `run`: Runs the given work item.
- `result`: Returns the result after running the function `fn`.
- `_condition`: Internal use only. Represents the condition variable that allows the task to be waited for.
- `_state`: Internal use only. Represents the current state of the execution.
- `_waiters`: Internal use only. Represents the list of threads currently waiting for the task’s completion.
- `cancel`: Cancels the task.
- `running`: Checks whether the task is currently running.
- `done`: Checks whether the task has completed execution.
- `__lt__`: Compares the priority of two `Task` objects or a `Task` object and an integer.

## Usage

You can use the `Task` class to manage and handle tasks in your Python scripts or programs. Here is an example of how to use this class:

```python
from taskmodule import Task

# Define a task
task = Task(fn=some_function, args=[arg1, arg2], kwargs={'kwarg1': val1, 'kwarg2': val2}, timeout=10, priority=1)

# Run the task
task.run()

# Get task result
result = task.result()

# Check if task is running
is_running = task.running()

# Check if task is done
is_done = task.done()
```

Note: `some_function`, `arg1`, `arg2`, `val1`, and `val2` should be defined or imported before creating the task.