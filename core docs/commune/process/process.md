## Readme

#### Description
This script contains a `Process` class that uses the `commune` and `multiprocessing` modules to manage multiple processes. It provides functionalities to create new processes, track their life span, stop them, and query their status.

#### Methods:

- **queue:**: Creates a multiprocessing.Queue object.

- **process:**: Adds a new Process to the process map and starts it if the `start` argument is `True`.

- **start:**: Alias for `process`.

- **getppid:**: Returns the parent process ID.

- **getpid:**: Returns the current process ID.

- **get_age:**: Returns the time since a given process started.

- **oldest_process_name:**: Returns the name of the longest running process.

- **oldest_process:**: Returns the longest running process.

- **oldest_pid:**: Returns the Process ID of the oldest process.

- **n:**: Returns the number of processes in the process map.

- **join:**: Stops and removes all processes from the process map.

- **stop_all:**: Alias for `join`.

- **stop:**: Stops a process and removes it from the process map.

- **remove_oldest:**: Stops and removes the oldest process from the process map.

- **fleet:**: Creates multiple processes with given arguments and adds them into the process map.

- **processes:**: Returns the list of names of all processes in the process map.

- **fn:**: Returns 1.

- **test:**: Runs a test that starts and stops n processes sequentially.

- **semaphore:**: Creates a semaphore object with a maximum value of n.

- **__delete__:**: Stops all processes and deletes the object.

#### Usage:
- Import the class: `from filename import Process`
- Start a new process: `Process.start(fn='function_name', args=[arg1, arg2], kwargs={'key':'value'}, tag='mytag', name='myprocess_name')`
- Get the oldest process information: `Process.oldest_process()`

This `Process` class provides a simplified interface to manage multiple processes, starting them, tracking their lifetime and removing them as necessary while ensuring each process does not hang. Each process can be identified by name, and the status of individual processes as well as the overall process pool can be queried.

Note that you need to have `commune` installed in your system to use this class.
