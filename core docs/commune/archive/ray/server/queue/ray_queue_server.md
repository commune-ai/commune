# Python Code Readme

This script includes the implementation of the classes `QueueServer`, `QueueClient`, and `RayActorClient`. The implementation is based on Ray's actor system to handle distributed operations/tasks in queue servers and clients.

## Key Classes and their methods:

1. `QueueServer`: Handles distributed queue system with basic operations, such as creating and deleting queue topics, listing queue topics, putting and retrieving items in a queue topic, and checking a queue's condition(s).

    - `topic2actorname`:  Returns the name of the actor associated with a topic.
    - `create_topic`:  Creates a new queue with a maximum size and assigns an actor to manage it.
    - `delete_topic`:  Deletes a queue topic and shuts down its actor.
    - `get_queue`: Gets a queue from storage by its topic name.
    - `list_topics`: Lists all existing topics.
    - `put` & `put_batch`: Puts an item or a batch of items in a queue topic, respectively.
    - `get` & `get_batch`: Gets an item or a batch of items from a queue topic, respectively.
    - `size`, `empty`, `full`: Returns the size of a queue, checks if the queue is empty, and checks if the queue is full, respectively.

2. `QueueClient`: Inherits from `QueueServer` and overloads some methods to fit the client operations.

    - `delete_topic`:  Deletes a queue topic and kills its managing actor.
    - `size`, `isempty` (also available as `empty`), `isfull` (also available as `full`): Returns the size of a queue, checks if the queue is empty, and checks if the queue is full, respectively.

3. `RayActorClient`: Allows interacting with a `Module` instance actor using common Python method calls.

   - Upon initialization, it appends methods of the module actor to itself and makes them available as local methods. It determines whether to return object IDs or results based on the `ray_get` argument.
   
The main components used in this script are `ray` (for distributed and parallel execution) and `commune` (for handling distributed systems). The script also includes a standalone main block which deploys a `QueueServer` actor and tests queue operations.

You can run the script via the command line using the command: `python script_name.py`. 

**Please Note**: This script might require knowledge of distributed system, parallel programming in Python using the ray and commune libraries.