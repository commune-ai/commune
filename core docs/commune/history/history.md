# Commune.History Module

The History module is part of the commune library designed to manage history related functionality for applications.

## Features

- Creating an instance of history based on a unique key
- Storing history item into the history path
- Retrieving history of a specific key
- Retrieving all history
- Removing history of a specific key
- Removing all history from the history path

## Usage

```python
history_instance = History(history_path='new_history')

history_instance.add_history({ 'epoch': '1', 'timestamp': '1' })

history = history_instance.history()
all_history_data = history_instance.all_history()

history_instance.rm_key_history(history_path='new_history')
history_instance.rm_history(history_path='new_history')
```

## Note

The key should be a unique identifier for each history instance. The history path is the directory in which all history are stored. If any error or exception happens, make sure you have the correct permissions and the directory exists.

The `ls` method lists all items under a directory, `put` method stores an item in a directory, `glob` method fetches all data from a directory, and `rm` method delete item(s) under directory.

## Install

This module is part of commune library, ensure that you installed commune properly. If not, you can install it using pip.

```shell
pip install commune
```