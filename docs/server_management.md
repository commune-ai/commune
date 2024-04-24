
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



Where teh args and kwargs are the positional and keyword arguments of the function and the timestamp is the timestamp of the request. The signature is the signature of the data using the user's private key.

```json
{
    "data" : "{'args': [], 'kwargs': {'a': 1}, 'timestamp': 1713911754.136804}",
    "signature" : "59rcj4fjdjiwjoijveoivjhowuhveoruvherouvhnerouohouev"
}

or 
```

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



