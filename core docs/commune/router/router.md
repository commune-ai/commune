# Router Module

This is a Python code that provides a concurrent thread pooling mechanism with a priority queue.

## Overview

The main class defined in the router module is `Router(c.Module)`. This class provides methods for managing and executing threads and tasks in a priority queue. Tasks are represented as future objects that can return results when done.

## Initialization

The `Router` class constructor takes in parameters such as `max_workers` to set the maximum number of threads that can execute the tasks, `maxsize` to set the maximum size of the queue (default to infinite if not specified), and `thread_name_prefix` to set an optional name prefix for the threads.

## Adding Tasks

Tasks can be added to the thread pool by using the `submit` method. It takes in parameters such as the module name, function name, and their respective arguments and dispatches them as tasks to execute.

## Managing Tasks

- The `adjust_thread_count` method is used to manage the available threads. If idle threads are available, no new threads are spun off. Otherwise, new threads are created and started.
- The method `shutdown` is used to stop the worker threads. If `wait` is set to `True`, it will also block until all worker threads have exited.

## Worker Threads

The main work of executing tasks is done inside the `worker` method. This method runs inside each worker thread and processes tasks from the task queue. If the executor is shut down, the worker thread will exit.

## Handling Futures

The `wait()` method is used to block until all futures are done and return their results. `as_completed()` method is a generator that yields futures as soon as they are completed.

## Test Method

The `test()` method is used to perform a basic test of the thread pooling functionality. It starts a server, submits a task to it, evaluates the result, and then kills the server.

## Dashboard Functionality

This also has a dashboard functionality that lets you monitor several parameters and activities of the thread pool. The `dashboard()` and `playground_dashboard()` methods support this functionality.

## Note

The code makes use of various Python standard libraries like `concurrent`, `threading`, and `queue`. It also uses the `commune` library for handling the server and the `loguru` library for logging.