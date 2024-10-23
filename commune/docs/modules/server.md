
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
c.restart('demo')  # Restart the module which will run back on the same port
```


