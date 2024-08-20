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



Endpoints

The default whitelist functions are the functions from the Module class. These functions are available to all users by default. The whitelist does not require a

These are functions that are available to all users by default.

['info', 'metadata', 'schema', 'server_name', 'is_admin', 'namespace', 'whitelist', 'endpoints', 'forward', 'fns']


c.module('module').endpoints()

info: the info of the module
endpoints : the endpoints of the module
schema : the schema of the module
metadata : the metadata of the module
namespace : the namespace of name -> address for the servers.
server_name : the name of the server
is_admin : checks whether the user is an admin


Stake Per Rate: Stake Based Rate Limiting

We do stake based rate limiting by having a stake2rate variable that determiens the number of calls per minute by dividing the total stake by the stake2rate variable.



The Access Module

The access module is a module that is used to restrict access to other keys based on the stake and user permissions. Commune has admins and users. Admins can call any function, while users can only call whitelisted functions. If the key is local, then the user can call whitelist functions at any rate. This can be disabled, but is not recommended as leaking any key information can be dangerous if any key on your local machine is compromised.



```python
1. Verify Address using the Signature and Data by the user's public key's private key

2. Verify the Timestamp staleness was within the last n seconds (2 seconds)

3. Verify the user identity, whether the function is callable (whitelisst or blacklisted) or if the user is an admin.

if user is admin
    return True
else:
    if function is in whitelist and not in blacklist:
        if rate < max_rate(user):
            return True
        else:
            return False
    else:
        return False

```


To see the module

c access/filepath # for the path
c access/code # for the code



