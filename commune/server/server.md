Access to A Module


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

w
