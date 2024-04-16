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

c.users()