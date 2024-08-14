The serializer is responsible for making sure the object is json serializable



The rules are simple

If the object is a dictionary, we iterate over the keys and values and serialize them. 
If the value is a dictionary, we recursively put that dictionary through the serializer
if the value is not a dictionary, we see if the value is json serializable. 

Default json serializable types are:
- str
- int
- float
- bool
- None


Adding a new type is simple. Just add the type to the `SERIALIZABLE_TYPES` list in the `Serializer` class.

If the value is not json serializable, we raise a `NotSerializableError` exception.

The serializer is used in the `Commune` class to serialize the object before it is saved to the database. 
```

```python
# File: commune/serializer/serializer.py
from typing import Any, Dict, Union
def serialize_{type}(obj: {type}) -> Dict[str, Any]:
    return {{"value": obj.value}}

def deserialize_{type}(data: Dict[str, Any]) -> {type}:
    return {type}(data["value"])
```

Now when that type is encoutered, the serializer will use the `serialize_{type}` and `deserialize_{type}` functions to serialize and deserialize the object.

