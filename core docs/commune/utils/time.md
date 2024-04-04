# Time Utility Functions and Timer Class

This module provides several utility functions for manipulating date and time in different formats. Additionally, it includes a Timer class for timing code execution.

## Functions:


### `get_current_time() -> str`

This function returns the current UTC time as a string in the format "%m%d%H%M%S".

### `roundTime(dt=None, roundTo=60)`

This function rounds a datetime object to the closest time lapse in seconds. By default, it uses the current time and rounds to the nearest minute. 

### `isoformat2datetime(isoformat:str) -> 'datetime.datetime'`

This function converts an ISO formatted string to a datetime object.

### `isoformat2timestamp(isoformat:str, return_type='int')`

This function converts an ISO formatted string to a timestamp of the specified type: either 'int' or 'float'. 

### `timedeltatimestamp( **kwargs)`

This function returns the difference between the current timestamp and a specified time in the past given in any of the following units: 'hours', 'seconds', 'minutes', 'days'. 

### `hour_rounder(t)`

This function rounds a datetime object to the nearest hour.


## Class:

### `Timer`

This class is used to time the execution of code blocks. It has methods to start and stop the timer, and it can be used as a context manager with the `with` statement. It also has a `seconds` property to get the elapsed time.

## Usage:

These functions and classes provide a toolkit for working with dates and times in Python, whether to convert between common formats, round or calculate differences in time, or time code execution. They are useful in a wide range of contexts that require time manipulation, particularly data analysis, logging and debugging.