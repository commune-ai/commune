# Bittensor Serializer

Bittensor Serializer is a Python library that provides serialization and deserialization functions for Bittensor tensors, supporting various data formats and types.

## Overview

This `Serializer` class majorly consists of two methods `serialize()` and `deserialize()`, that are used to convert complex data types into a format that can be easily stored or transferred and then reconstructing them. These methods support different data types like Python's inherent dictionaries, lists, sets, tuples, etc., along with other complex data types like torch tensors and numpy's ndarrays. 

Moreover, the `serialize()` function can convert a Python dictionary object into string and bytes formats based on the mode specified. And the `deserialize()` function can convert a string and bytes representation of a Python dictionary back into the original dictionary.

## Getting Started

### Prerequisites

To use this library, you need to have following Python packages installed. 

```
import numpy as np
import torch
import msgpack
import msgpack_numpy
from typing import Tuple, List, Union, Optional
from copy import deepcopy
from munch import Munch
import commune as c
import json
```

### Function Details:

-   `serialize(self,x:dict, mode = 'str')`

    This function is used to perform serialization of data which can be a dict, list, set, tuple etc. The serialization mode can be set to 'str' or 'bytes.'

-   `deserialize(self, x:dict, mode = 'str')`

    This function is used to perform deserialization of data which can be a dict, list, set, tuple etc. The deserialization mode can be set to 'str' or 'bytes.'

-   `resolve_value(self, v)`

    This function is used to resolve the values within data structures. This can be used while serializing complex data structures.

-   `is_serialized(self, data)`

    This function checks if the given data is serialized or not.

-   `get_type_str(self, data)`

    This function is used to get the string representation of the data type of the input. 

### Testing

To ensure correct functionality, you can run two test functions provided, `test_serialize()` and `test_deserialize()`.

## License

This project follows the "I DONT GIVE A FUCK" license (IDGAF). Feel free to look, modify, or distribute this code as you wish. The author will not be responsible for any issues arising due to this code.