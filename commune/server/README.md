
## Serving


A server is a module that is converted to an http server.

To deploy a server

```python
# c serve model.openai is also vaid without the tag
# if you want to add a tag to the server
c serve model.openai:
# c.serve('model.openai')
```
{
    'success': True,
    'name': 'model.openai',
    'address': '162.84.137.201:50191',
    'kwargs': {}
}

The server can be called in several ways

```python
c call model.openai/forward "sup"
c.call("model.openai:/forward",  "sup")
```

"hey there homie"

The name of the endpoing is formated as

{server_ip}:{server_port}/{function}

with the data being a json request in the format of the following



### Viewing Available Servers
You can view the available servers using the `servers()` method:

```
c servers # c.servers()  
```
### Viewing Server Logs
To view the logs of a served module, you can use the `logs()` method:

```python
c logs demo c.logs('demo')
```

### Connecting to a Served Module
You can connect to a served module using the `connect()` method:
The following calls the `info()` function of the `demo:1` module:

```python
c call demo/info 
```

### Restarting a Module
You can restart or kill a served module using the `restart()` and `kill()` methods:

```python
c.restart('demo')  # Restart the module which will run bxack on the same port
```

SERIALIZER

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


TESTS

to test if commune is running properly 

c test

to test a module 

c test {modulename}


c core_modules is the core modules

