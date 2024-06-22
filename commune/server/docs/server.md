A server is a module that is converted to an http server.


The server can be called in several ways

The name of the endpoing is formated as


{server_ip}:{server_port}/{function}

with the data being a json request in the format of the following


```
{
    'data': '{"args": [], "kwargs": {}, "ip": "162.84.137.201", "timestamp": 
1713912586}',
    'crypto_type': 1,
    'signature': 
'3e429b1fa51dfbb3e15f9be00930c9a9b9086a803ce3b157f53c9a2525685201c6936cbc899b29a8cba30091db1a0a4876a86641f1e332c0614039080189ac87',
    'address': '5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC'
}
```

or 

```json
{
    "data" : "{'args': [], 'kwargs': {'a': 1}, 'timestamp': 1713911754.136804}",
    "signature" : "59rcj4fjdjiwjoijveoivjhowuhveoruvherouvhnerouohouev"
}

or 
```




