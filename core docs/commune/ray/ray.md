# Ray Module

The `Ray` module is a class that contains class methods and static methods for managing Ray clusters. It provides functionality to start, stop and restart the clusters, interact with the Ray runtime context, and handle Ray tasks and actors.

## Dependencies
Ray module depends on many libraries, some of them are:
- commune
- ray
- json
- torch

## Features

### Initialize Ray environment

The `init` method initializes the Ray environment with given configuration. If Ray environment is already initialized, it retrieves the Ray runtime context.

### Start, Stop & Restart Cluster

`start`, `stop` and `restart` methods are used to manage the Ray cluster.

### Ray runtime context

Functions like `ray_context`, `get_runtime_context` and `ensure_ray_context` provide ways to work with the ray runtime context. 

### Interact with Ray tasks

`ray_tasks`, `ray_wait` and `ray_get` methods manage Ray tasks. 

### Interact with Ray actors

The `Ray` class includes various methods to interact with Ray actors, including `ray_put`, `get_actor`, `kill_actor`, `actor_exists`, `actors`, `list_actors`, `actor2id`, `id2actor`, and `get_actor_id`.

### Resource Management

`Ray` module also provides features for resource management through `actor_resources` method.

## Examples

To use the `Ray` module, first import the necessary libraries, initialize the Ray context, and then use the class methods provided according to your needs.
```python
# import libraries
import commune as c
import ray 

# initialize Ray context
c.Ray.init()

# start Ray cluster
c.Ray.start()

# get Ray runtime context
context = c.Ray.get_runtime_context()

# stop Ray cluster
c.Ray.stop()
```