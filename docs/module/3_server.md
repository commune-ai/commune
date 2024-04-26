
## Serving


A server is a module that is converted to an http server.

To deploy a server
c serve model.openai::tag

{
    'success': True,
    'name': 'model.openai::tag',
    'address': '162.84.137.201:50191',
    'kwargs': {}
}

The server can be called in several ways

c call model.openai::tag/forward "sup"

"hey there homie"

The name of the endpoing is formated as

{server_ip}:{server_port}/{function}

with the data being a json request in the format of the following



### Viewing Available Servers
You can view the available servers using the `servers()` method:

```python
c.servers()
```

### Viewing Server Logs
To view the logs of a served module, you can use the `logs()` method:

```python
c.logs('demo::tag1', mode='local')
```

### Connecting to a Served Module
You can connect to a served module using the `connect()` method:
The following calls the `info()` function of the `demo::tag1` module:

```python
c.connect('demo::tag1').info()
```

### Restarting and Killing a Served Module
You can restart or kill a served module using the `restart()` and `kill()` methods:

```python
c.restart('demo::tag1')  # Restart the module
c.kill('demo::tag1')     # Kill the module
```

---


Where teh args and kwargs are the positional and keyword arguments of the function and the timestamp is the timestamp of the request. The signature is the signature of the data using the user's private key.

```json
{
    "data" : "{'args': [], 'kwargs': {'a': 1}, 'timestamp': 1713911754.136804}",
    "signature" : "59rcj4fjdjiwjoijveoivjhowuhveoruvherouvhnerouohouev"
}



Verification


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


Whitelist/Blacklist Functions

If you want to restrict access to a module, you can use the whitelist and blacklist functions. These functions are used to restrict access to a module based on the user's role.

Only Admins can have access to all functions in a module and there is by default one admin per commune (computer). The admin can add other admins if needed.

Anyone who is not the admin can be assigned a role. They also must only call functions that are whitelisted for their role.

To add a user to a role, use the add_user function. For example, to add a user to the admin role:

```python
c.add_user("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY", role="admin")
```

To add a user to a custom role:

```python
c.add_user("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY", role="custom_role")
```


Default Whitelist Functions

The default whitelist functions are the functions from the Module class. These functions are available to all users by default. The whitelist does not require a blacklist of any functions as at the moment.


c.module('module').whitelist

whitelist = ['info',
            'schema',
            'server_name',
            'is_admin',
            'namespace',
            'whitelist', 
            'blacklist',
            'fns'] # whitelist of helper functions to load


blacklist = []


Stake Per Rate: Stake Based Rate Limiting

We do stake based rate limiting by having a stake2rate variable that determiens the number of calls per minute by dividing the total stake by the stake2rate variable.


To see the module

c access/filepath # for the path
c access/code # for the code


Admins

Admins are the ones who can do anything on a computer and from outside a computer. They have all access to the commune's functions and can add other admins. There is by default one admin per commune. Admins can add other admins if needed.

c.add_admin("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY")

or 

c.add_user("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY", role="admin")


To add a custom role

c.add_user("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY", role="custom_role")


To remove an admin, use the remove_admin function.

c.rm_admin("5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY")

To see all users

{
    '5CfWRdKjT5cUjSnpZA7xuW3a3qsXWkisbqrCrBky3L12Wc8R': {'role': 'admin'},
    '5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC': {'role': 'admin'},
    '5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY': {'role': 'admin'}
}

To see the user2role mapping

c user2role
{
    '5CfWRdKjT5cUjSnpZA7xuW3a3qsXWkisbqrCrBky3L12Wc8R': 'admin
    '5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC': 'admin',
    '5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY': 'admin'
}



To get the roles per users


c role2users

{
    'admin': ['5CfWRdKjT5cUjSnpZA7xuW3a3qsXWkisbqrCrBky3L12Wc8R', '5GZBhMZZRMWCiqgqdDGZCGo16Kg5aUQUcpuUGWwSgHn9HbRC', '5DUdqShkPzzVDYGznZJQ92xP8cdp5BTt9XSrgMDMgHRArRyY']
}



