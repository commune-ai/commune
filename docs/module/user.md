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