# Ray Python Code

This Python script integrates the Ray library for distributed computing, helping to manage tasks and actors in a distributed system.

## Functions

- `limit_parallel_wait`: This function manages the number of parallel jobs running in the system. It stops adding new jobs when the parallel limit is reached and resumes when jobs are completed.

- `kill_actor`: This function kills or removes a specified actor from the distributed system.

- `actor_exists`: This function checks if a specific actor exists in the distributed system.

- `create_actor`: This function creates a new actor with the provided specifications including its name, resources and concurrency limit. It also includes options for refreshing and making redundant copies of the actor.

- `custom_getattr`: This function allows to retrieve attributes from nested objects by providing the attribute path as a string, supporting both dictionary syntax and attribute access syntax.

## RayEnv Class

The `RayEnv` class manages the initialization and shutdown of the Ray context. This allows using the `with` syntax to automatically handle the setup and teardown of the Ray environment.

It includes the `is_initialized` property which checks whether the Ray context has already been initialized, and properties for controlling the context entry and exit points (`enter_context_gate`, `__enter__`, `__exit__`). 

## Dependencies

- Ray : Open Source, Fast, Scalable and User Friendly Distributed Computing.
- Torch: Scientific Computing package - Tensors and Dynamic neural networks in Python with strong GPU acceleration.

## Usage 

This Python script is importable. You can import it in your Python program and utilize the functions and class for managing tasks and actors in a distributed system using Ray and Torch.
