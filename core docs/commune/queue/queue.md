## Readme

#### Description
`QueueServer` is a simple and powerful Python class based on `commune` and `asyncio` for managing multiple queues concurrently. This class provides functionalities to easily create, fill, retrieve data from queues while maintaining concurrency. 

#### Class Initialization Parameters:

- **max_size** (int): Maximum size of a Queue. Default is 1000.
- **mode** (str): The mode/technology to implement the queue. Default is 'asyncio'
- **kwargs**: Other optional keyword arguments.

#### Methods:

- **queue_exists**: Checks if a queue with a given key exists.

- **add_queue**: Adds a new queue with the specified key.

- **get**: Returns the next item from the queue.

- **put**: Puts an item into the queue.

- **get_batch**: Retrieves a set number of items from the queue as a batch.

- **size**: Returns the current size of a queue.

- **empty**: Checks if a given queue is empty.

- **full**: Checks if a given queue is full.

- **size_map**: Returns a dictionary representing the size of each queue.

- **test**: Test method to check the functionality of the `QueueServer`. It creates multiple queues, populates with data and retrieves it while asserting the size of the queue throughout the operations.

#### Usage:

- Import the class: `from filename import QueueServer`
- Initialize a `QueueServer` instance: `qs = QueueServer()`
- Add a new queue: `qs.add_queue('queue_key')`
- Put an item into the queue: `qs.put('queue_key', 'item')`
- Get the next item from the queue: `qs.get('queue_key')`

This `QueueServer` class provides a simplified interface for managing multiple queues and conducting concurrent operations on them. Its asynchronous operations ensure that your program remains highly efficient, regardless of the number of queues or the amount of data. Through its various APIs, it provides complete control over the data management of queues.