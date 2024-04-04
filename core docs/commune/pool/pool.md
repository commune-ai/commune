## Readme

#### Description
This script contains a `Pool` class that wraps Python's multiprocessing module to ease the task of running multiple processes in parallel. It provides an API that includes creating named worker processes, checking their status, stopping them, and scaling the process pool up and down. This class also handles communication between the main and child processes through a multiprocessing.Queue object.

#### Methods:

- **__init__:**: Creates the Pool instance by initiating the required number of replicas and a queue.

- **add_replica:**: Adds a new running process to the process pool. 

- **num_replicas:**: Returns the number of running processes in the process pool. 

- **pop_replica**: Removes a process from the process pool.

- **start:**: Starts a new process in the process pool.

- **getppid:**: Returns the parent process ID.

- **getpid:**: Returns the current process ID.

- **get_age:**: Returns the time since a given process started running.

- **oldest_process_name:**: Returns the name of the longest-running process in the pool.

- **oldest_process:**: Returns the longest-running process in the pool.

- **oldest_pid:**: Returns the Process ID of the oldest process.

- **n:**: Returns the number of processes in the process pool.

- **join:**: Stops and removes all processes in the pool.

- **stop_all:**: Alias for `join`.

- **stop:**: Stops a process and removes it from the pool.

- **remove_oldest:**: Stops and removes the oldest process in the pool.

- **fleet:**: Creates multiple processes and adds them into the process pool.

- **processes:**: Returns the list of names of all processes in the pool.

- **fn:**: Returns 1.

- **test:**: Runs a test that starts and stops n processes sequentially.

- **semaphore:**: Creates a semaphore object with a maximum value of n.

- **__delete__:**: Removes all processes in the pool upon deletion.

#### Usage:
- Import the class: `from filename import Pool`
- Create an instance with n replicas: `pool = Pool(module='mod_name',replicas=3)`
- Call a method to control the processing pool: `pool.add_replica(module='mod_name2')`

This pool manager provides a simple way to manage multiple processes, adding or removing them as necessary and ensuring that no process remains hanging. Each process in the pool can be identified by name, and the main program can query the status of individual processes as well as the overall pool.

Note that this class uses the 'commune' package (aliased as `c`). You need to have `commune` installed in your system to use this class.