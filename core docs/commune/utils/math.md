# Python Classes for Running and Moving Window Averages

This Python module provides two classes, `RunningMean` and `MovingWindowAverage`, for calculating the running and moving window averages of a sequence of numbers.

## Class Descriptions:

### `RunningMean`
This class calculates the running mean of a sequence of numbers. It maintains the total value and count of the numbers seen so far, and returns the current average when needed.

Functions:
- `__init__(self, value=0, count=0)`: Initializes the `RunningMean` object.
- `update(self, value, count=1)`: Updates the total value and count with the new value and count.
- `value(self)`: Returns the current average.
- `to_dict(self)`: Returns a dictionary representation of the object.
- `from_dict(self, d: Dict)`: Updates the object's attributes from a dictionary.

### `MovingWindowAverage`
This class calculates the moving window average of a sequence of numbers. It maintains a window of the most recent numbers and calculates the average of the numbers in this window.

Functions:
- `__init__(self,value: Union[int, float] = None, window_size:int=100)`: Initializes the `MovingWindowAverage` object.
- `set_window(self,value: Union[int, float] = None, window_size:int=100)`: Sets the window size and window values.
- `update(self, *values)`: Updates the window values with the new values.
- `value(self)`: Returns the current average.
- `to_dict(self)`: Returns a dictionary representation of the object.
- `from_dict(self, d: Dict)`: Updates the object's attributes from a dictionary.
- `test(cls)`: A class method for testing the `MovingWindowAverage` class.


## Utility Function:

### `round_sig(x, sig=6, small_value=1.0e-9)`
This utility function rounds a given number `x` to the specified number of significant digits.

## Usage:

These classes are useful for situations where the average of a sequence of numbers is required, and the sequence is too large to store in memory, or the average needs to be updated in real-time as new numbers arrive. For `MovingWindowAverage` class, it also provides window-size based control over the sequence's most recent numbers.