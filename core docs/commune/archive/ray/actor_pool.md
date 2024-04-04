# ActorPool

This code defines the `ActorPool` class in [Ray](https://docs.ray.io/en/latest/). Ray is a high-performance framework for distributed computing applications, and this class provides several utilities to manage a pool of actors in Ray.

## Class Definition

The `ActorPool` class has the following methods:

- `__init__(self, actors: list)`: This method is the constructor and it initializes the pool with a given list of actor handles.

- `map(self, fn: Callable[[Any], Any], values: List[Any])`: This method takes a function and a list of values and applies the function to the actors in the pool and the values in parallel. 

- `map_unordered(self, fn: Callable[[Any], Any], values: List[Any])`: This method works similar to `map` but it returns the results as they become available rather than in the order of submission.

- `kill_idle(self)`: This method kills all the idle actors in the pool.

- `submit(self, fn, value)`: This method submits a single task to the pool to be run on the next available actor.

- `has_next(self)`: This method checks whether there are any pending results to return.

- `get_next(self, timeout=None, ignore_if_timedout=False)`: This method retrieves the next available result from the tasks that have been submitted to the pool.

- `get_next_unordered(self, timeout=None, ignore_if_timedout=False)`: This method works like `get_next` but it retrieves any available result rather than the next one in line.

- `has_free(self)`: This method checks whether there are any idle actors in the pool that are not currently running tasks.

- `pop_idle(self)`: This method removes and returns an idle actor from the pool if one is available.

- `push(self, actor)`: This method adds an actor to the pool.

## Usage

The `ActorPool` class is designed to facilitate concurrent execution of tasks across multiple actors in Ray. You can use this class to manage your actors and execute tasks efficiently.

Please note that this class is deprecated and the recommended way is to use either `ray.util.multiprocessing` for stateless/task processing or `Datasets.map_batches` for stateful/actor processing such as batch prediction. Detailed information can be found in the deprecation message in the class's docstring.