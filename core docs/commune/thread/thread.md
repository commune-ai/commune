# README - Thread Module

The Thread module is a Python-based functionality of the Commune library that simplifies multi-threading. It is designed to spawn and manage multiple threads in an efficient and easy manner. 

## Key Functions

### Thread creation: 

Class method `Thread.thread()` is utilized to spawn a thread.

```python
t = Thread.thread(fn=my_function, args=['sample_arg'], kwargs={'key': 'value'}, daemon=True, tag='my_thread')
```

The `daemon` parameter is optional and you can also set `start = False` to prevent the thread from immediately starting after creation.

### Thread management

- **Queue:** Class method `Thread.queue()` is used to create a queue with a maximum size. If the maximum size is not specified or set to zero, then queue size is infinite.

    ```python
    q = Thread.queue(maxsize=10)
    ```
- **Join Threads:** Class method `Thread.join_threads()` stops the execution of the calling thread until the thread(s) specified in the method's argument finishes their task.

    ```python
    Thread.join_threads(['thread1', 'thread2'])
    ```
- **Threading Fleet:** Class method `Thread.thread_fleet()` creates a number of threads executing the same function.

    ```python
    threads = Thread.thread_fleet(fn=my_function, n=5) 
    ```
- **Threads:** Class method `Thread.threads()` returns the name of all the active threads.

    ```python
    all_threads = Thread.threads()
    ```
- **Number of Threads:** Class method `Thread.num_threads()` returns the number of active threads.

    ```python
    num_threads = Thread.num_threads()
    ```
- **Semaphore:** The `semaphore` method is used to limit the number of threads concurrently executing.

    ```python
    s = threading.Semaphore(n=5)
    ```

## Usage:

Here are the main steps you should follow when using the Thread module,

1. Import the 'Thread' class from the 'commune' library.
2. Instantiate the Thread class.
3. Implement the main functionalities such as spawning threads, joining threads, creating queuing, and managing threads.
