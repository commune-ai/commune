
## Serving

You can serve a module to make its functions accessible via a server.

### Serving a Module
You can serve a module using the `serve()` method, optionally providing a tag for versioning:

```python
demo.serve(tag='tag1')
```

### Viewing Available Servers
You can view the available servers using the `servers()` method:

```python
c.print(c.servers())
```

### Viewing Server Logs
To view the logs of a served module, you can use the `logs()` method:

```python
logs = c.logs('demo::tag1', mode='local')
c.print(logs)
```



### Connecting to a Served Module
You can connect to a served module using the `connect()` method:

```python
demo_client = c.connect('demo::tag1')
demo_client.info()
```

### Restarting and Killing a Served Module
You can restart or kill a served module using the `restart()` and `kill()` methods:

```python
c.restart('demo::tag1')  # Restart the module
c.kill('demo::tag1')     # Kill the module
```

---

