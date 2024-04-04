# Readme for Code "Priority Thread Pool Executor"

This library offers a factory method implementation for creating a priority thread pool in Python. The Python code basically uses two main classes, ThreadRipper and PriorityThreadPoolExecutor, for managing threads in the pool based on task priority. It was created in 2021 by Yuma Rao under the MIT License.

## Class Breakdown

### 1. ThreadRipper

ThreadRipper serves as a factory method for creating a priority thread pool. It has class methods for handling commandline arguments, checking the configuration and adding defaults from the environment variables. Its `__new__` method creates a new priority threadpool with the option to define the number of maximum workers and the maximum number of tasks in the priority queue.

### 2. WorkItem & worker

These are internal classes used for handling work items within the thread pool. WorkItem manages the execution of a task, handling any possible exceptions. worker is the actual work executor that picks up work items from the priority queue and executes them.

### 3. PriorityThreadPoolExecutor

A key class in the code which takes the priority of the task as an argument and manages the execution of tasks based on their priorities. This class manages the worker threads and job queueing as priority. It submits the job to the worker and adjusts thread count based on the current state of the thread pool. It also handles the shutdown of the executor.

## Usage

To use this thread pool library, first create an instance of the `ThreadRipper` with your desired settings. You can specify the maximum number of workers, maximum size of tasks, and the configuration.

After that, you can use `PriorityThreadPoolExecutor` to submit your tasks. Remember to assign the priority to your tasks, the tasks with higher priority will be run prior to lower ones.

Finally, when all tasks are finished, call `shutdown()` method to clean up the threads.

## Requirements

This library requires Python 3.6 or higher. It also utilizes the following modules: os, argparse, copy, bittensor, concurrent.futures, itertools, queue, random, threading, weakref, time, loguru. It should be noted that 'bittensor' is not a standard Python library and may need to be manually installed or substituted.
