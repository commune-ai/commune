# Remote Module README

The purpose of this Python module is to ease the management of remote hosts with SSH through a clean and extensible API. 

## Key Features
- Add and remove hosts to oversee, with properties such as host IP address, port, username, and password.
- Execute commands or scripts remotely across all managed hosts.
- Easily switch the host management file depending on the working context.
- Automatically generate and manage SSH keys on hosts.
- Obtain valuable metadata about hosts, such as CPU, memory, and disk usage.
- More advanced features aimed at managing a fleet of servers, such as serving a function, checking if a host is an admin, and rotating peers.

## Typical Usage
```python
from remote import Remote

# Create a new manager object
manager = Remote()

# Add a new host
manager.add_host(host='192.168.1.1', port=22, user='root', pwd='password')

# Execute command across all hosts
manager.cmd('ls -l', verbose=True, sudo=True)

# Remove a host
manager.rm_host('root10')

# Get a map of all hosts with their corresponding SSH connection strings
ssh_map = manager.host2ssh()

# Obtain a list of all hosts which meet the search criteria
filtered_hosts = manager.filter_hosts(include='module', avoid='test')
```

Please refer to the code and comments therein, as they contain a lot of valuable information on how to use this class more effectively. Some functions may require additional dependencies for fringe cases, which are handled gracefully by attempting to import those dependencies only when those functions are called.

## Note:
- Please ensure that SSH login with password is enabled on the hosts for the username you provide.
- Please ensure that the user has necessary privilege to execute the commands you provide.
- This class is not intended for highly secure operations. Be careful about password management and consider securing communication using SSH keys where necessary.
- Remember to have SSH installed and configured correctly in your working environment.