# Readme

This script provides two classes, `CustomThread` and `ThreadManager`, to control threads in Python.

## Prerequisites

- Python 3
- `threading` module installed

## Code Explanation

- `CustomThread` class extends `threading.Thread` class. It implements a custom constructor (__init__) for initializing a thread with certain function (fn), arguments (args) & keyword arguments (kwargs). It also allows you to specify if you want the thread to run forever.
- `run` method inside the `CustomThread` class is the target function of the thread class. It executes the passed function with the given arguments. If the thread is set to run forever, it keeps executing the target function.
- `stop` method can be used to stop the thread execution.
- Method `get_id` returns the id of the respective thread.

- `ThreadManager` class has been designed to handle and manage multiple threads.
- In the `__init__` or the constructor, it initializes the max_threads, sessions and threads.
- `submit` method is used to submit a new function to be run in a new thread, and this thread is added to the list of threads.
- `threads` property returns a list of all the threads.
- `__del__` method is used to automatically shutdown all threads when an instance of the ThreadManager class is deleted.
- Method `shutdown` is provided to stop all running threads manually.

## Quick Start

You can use the script as follows:

```python
from time import sleep

# function to be run in a thread
def my_function(arg1, arg2):
    sleep(1)
    print(arg1, arg2)
    
# Create thread by passing the function and arguments
thread = CustomThread(fn=my_function, args=[1, 2])

# Start the thread
thread.start()

# Now, create a thread manager.
threadManager = ThreadManager()

# Submit a function to be run in a new thread
threadManager.submit(fn=my_function, args=[3, 4])
```

## Note

The `CustomThread` runs a function indefinitely in a thread unless 'forever' argument is set to False or the `stop` method is called.
The `ThreadManager` class helps manage multiple threads in an effective manner. Please ensure to handle exceptions within threads to prevent them from reaching an unhandled state and making your application unstable. Also, remember to use safe data structures when working with multithreading.
