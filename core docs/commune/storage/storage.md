# Storage Module

This Python module, named `Storage`, provides classes and methods for handling storage-related tasks in a distributed setting. It includes the capability to store and retrieve data, validation of data, handling replicas and shards of data, and providing dashboard functionalities in Streamlit.

## Classes

### Storage

This is the main class in the module which provides the functionalities for storage.

## Methods

The module provides the following methods among others:

- `put`: Store a new item in storage.
- `get`: Retrieve an item from storage.
- `exists`: Checks whether an item exists in the storage.
- `remove`: Remove an item from storage.
- `validate`: Validate the correctness of an item in the storage.
- `dashboard`: A Streamlit interface to interact with the methods of the Storage class.
- `refresh` : Removes all the items from the storage.
- `test` :  A method to test the basic functionality of the module.
- `get_shards` : Retrieve the shards of a specific data item.
- `put_dummies` : This is a function to store some dummy elements in storage.

## Attributes

Some important attributes used in Storage class are:

- `whitelist`: A list of functions that are allowed to be executed.
- `max_replicas`: Maximum number of replicas allowed for an item.
- `network`: The type of network being used. Can be local or remote.
- `max_shard_size`: The maximum size allowed for a shard.
- `min_check_interval`: The minimum interval that should pass before a check is performed.

Note: You need to have the 'commune' package installed in your python environment, before you can run this code.

## How to use this module

Please initialize the `Storage` class by providing the necessary parameters. Then call the relevant method as per your requirement.
