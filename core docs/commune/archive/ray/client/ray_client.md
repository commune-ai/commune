# Python Code Readme

This Python code is a ClientModule class implementation using Ray for parallel and distributed Python, extending the functionality of a Module from the Commune package. Ray allows you to scale your applications and to leverage parallelism. 

## Key methods and properties in the `ClientModule` class include:

- `__init__`: This constructor initialises the ClientModule instance, sets and parses actor if provided.

- `set_actor`: This method sets up the actor for the ClientModule. An actor can be identified by a string, a dictionary or an instance of `ray.actor.ActorHandle`.

- `getattr` and `setattr`: These two methods allow you to set and get attributes of the actor. In both methods, attribute values can either be directly returned or returned as an object ID, based on the flag `ray_get`.

- `submit`: This method calls the named function on the actor with given arguments and keyword arguments.

- `submit_batch`: This method allows you to concurrently perform a function call with a batch of arguments and keyword arguments on the actor.

- `remote_fn`: This method builds upon `ray.remote`, facilitating the distributed execution of a named function on the actor.

- `parse`: This method builds a function signature map where every function available on the actor is mapped to a partial function call with `client.remote_fn`, this makes functions available for remote execution.

- `__getattribute__`: This is a magic method overridden to handle attribute access for the ClientModule.

- `__setattr__`: This is a magic method overridden to handle attribute assignment for the ClientModule.


The main components used in the class are:

- Streamlit: Framework for Machine learning and Data Science.
- Ray: Library for parallel and distributed Python.
- Commune Package: Package that helps managing tasks within a distributed system.

**Please Note**: This code requires a solid understanding of distributed systems, parallel programming and magic methods. It is designed to utilize parallel processing across a network or a large system.

In the guard clause at the end, an instance of the `ClientModule` is created and deployed as an actor. This acts as a main driver program for testing the functionality of the ClientModule class.
