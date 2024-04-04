# RunningMean and MovingWindowAverage

This Python module offers two classes, `RunningMean` and `MovingWindowAverage`, which helps compute the running mean and moving window average of sequences of numbers respectively. 

## Class Definitions:

### `RunningMean`

- `__init__(self, value=0, count=0)`: Initializes a `RunningMean` object and sets the initial total value and count.

- `update(self, value, count=1)`: Increases the total value and count based on the supplied value and count.

- `value`: Returns the running mean if count is non-zero, otherwise returns infinity.

- `__str__(self)`: Allows the class object to be printed directly.

- `to_dict(self)`: Returns a dictionary containing all the attributes of the instance.

- `from_dict(self, d: Dict)`: Loads attribute values from a dictionary.

### `MovingWindowAverage`

This class manages a window of recent values and calculates an average of these values.

- `__init__(self,value: Union[int, float] = None, window_size:int=100)`: Initializes a `MovingWindowAverage` object.

- `set_window(self,value: Union[int, float] = None, window_size:int=100)`: Sets up the size of the window and the initial value inside that window. 

- `update(self, *values)`: Appends new values to the window's list and recalculates the average.

- `__str__(self)`: Allows the class object to be printed directly.

- `to_dict(self)`: Returns a dictionary containing all the attributes of the instance.

- `from_dict(self, d: Dict)`: Loads attribute values from a dictionary.

- `to_json(self)`: Returns a JSON representation of the object's state.

- `from_json(self, json_str:str)`: Restores the object's state from a JSON representation.

- `state_dict(self)`: Returns a dictionary containing the current state of the instance.

- `test(cls)`: A test method.

## Utility Function
### `round_sig(x, sig=6, small_value=1.0e-9)`

This utility function is used for rounding off a number `x` to a specific number of significant digits. 

## Usage:

These classes are useful for computing the running or moving mean of a sequence of numbers. This can be handy when dealing with large numbers or when dealing with continuously updated sequences. It also offers JSON serialization which might be useful for saving object states or sending across networks.